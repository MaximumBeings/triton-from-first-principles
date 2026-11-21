"""
Chapter 21 standalone: the Flash Attention core loop, causal masking inside
that loop (plus a correct-kernel-wrong-task demonstration), and top-2 MoE
gating (plus its renormalization trap).
"""
import torch
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Section 21.1 / 21.2: Flash Attention, non-causal and causal
# ---------------------------------------------------------------------------

@triton.jit
def flash_attn_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                       M, N, D,
                       stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                       stride_om, stride_od,
                       scale,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_idx = tl.arange(0, BLOCK_D)
    row_mask = rows < M
    d_mask = d_idx < D
    q_ptrs = q_ptr + rows[:, None] * stride_qm + d_idx[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)
    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        k_ptrs = k_ptr + cols[:, None] * stride_kn + d_idx[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0)
        v_ptrs = v_ptr + cols[:, None] * stride_vn + d_idx[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        scores = tl.where(col_mask[None, :], scores, float("-inf"))
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v, input_precision="ieee")
        m_i = m_new
    acc = acc / l_i[:, None]
    out_ptrs = out_ptr + rows[:, None] * stride_om + d_idx[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & d_mask[None, :])


@triton.jit
def flash_attn_causal_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                       M, N, D,
                       stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                       stride_om, stride_od,
                       scale,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_idx = tl.arange(0, BLOCK_D)
    row_mask = rows < M
    d_mask = d_idx < D
    q_ptrs = q_ptr + rows[:, None] * stride_qm + d_idx[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)
    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    max_col = pid_m * BLOCK_M + BLOCK_M
    for n0 in range(0, tl.minimum(N, max_col), BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        k_ptrs = k_ptr + cols[:, None] * stride_kn + d_idx[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0)
        v_ptrs = v_ptr + cols[:, None] * stride_vn + d_idx[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        causal_mask = rows[:, None] >= cols[None, :]
        valid = col_mask[None, :] & causal_mask
        scores = tl.where(valid, scores, float("-inf"))
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v, input_precision="ieee")
        m_i = m_new
    acc = acc / l_i[:, None]
    out_ptrs = out_ptr + rows[:, None] * stride_om + d_idx[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & d_mask[None, :])


def _run_attn(kernel, q, k, v, BLOCK_M=16, BLOCK_N=16):
    M, D = q.shape
    N, _ = k.shape
    BLOCK_D = max(16, triton.next_power_of_2(D))
    scale = 1.0 / math.sqrt(D)
    out = torch.empty((M, D), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M),)
    kernel[grid](q, k, v, out, M, N, D,
                 q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                 out.stride(0), out.stride(1), scale,
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)
    return out


def section_21_1_and_2():
    print("=== Section 21.1: Flash Attention core loop vs. naive full attention ===")
    torch.manual_seed(0)
    M, N, D = 17, 23, 8
    q = torch.randn(M, D)
    k = torch.randn(N, D)
    v = torch.randn(N, D)
    scale = 1.0 / math.sqrt(D)
    oracle = torch.softmax(q @ k.T * scale, dim=-1) @ v
    out = _run_attn(flash_attn_kernel, q, k, v)
    print("allclose to naive full attention?", torch.allclose(out, oracle, atol=1e-4))

    print()
    print("=== Section 21.2: causal Flash Attention, and a correct kernel on the wrong task ===")
    torch.manual_seed(0)
    M2 = N2 = 20
    D2 = 8
    q2 = torch.randn(M2, D2)
    k2 = torch.randn(N2, D2)
    v2 = torch.randn(N2, D2)
    scale2 = 1.0 / math.sqrt(D2)
    scores_full = q2 @ k2.T * scale2
    causal_mask = torch.tril(torch.ones(M2, N2, dtype=torch.bool))
    oracle_causal = torch.softmax(scores_full.masked_fill(~causal_mask, float("-inf")), dim=-1) @ v2
    oracle_noncausal = torch.softmax(scores_full, dim=-1) @ v2

    out_causal = _run_attn(flash_attn_causal_kernel, q2, k2, v2)
    print("causal kernel allclose to causal oracle?", torch.allclose(out_causal, oracle_causal, atol=1e-4))

    out_noncausal = _run_attn(flash_attn_kernel, q2, k2, v2)
    print("non-causal kernel allclose to CAUSAL oracle?", torch.allclose(out_noncausal, oracle_causal, atol=1e-4))
    print("non-causal kernel allclose to NON-causal oracle?",
          torch.allclose(out_noncausal, oracle_noncausal, atol=1e-4))


# ---------------------------------------------------------------------------
# Section 21.3: MoE top-2 gating
# ---------------------------------------------------------------------------

@triton.jit
def moe_top2_gating_kernel(logits_ptr, weight0_ptr, weight1_ptr, idx0_ptr, idx1_ptr,
                            M, E, stride_m, BLOCK_E: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_E)
    mask = cols < E
    row_ptr = logits_ptr + row * stride_m
    logits = tl.load(row_ptr + cols, mask=mask, other=float("-inf"))
    row_max = tl.max(logits, axis=0)
    p = tl.exp(logits - row_max)
    p = tl.where(mask, p, 0.0)
    probs = p / tl.sum(p, axis=0)
    top1_val = tl.max(probs, axis=0)
    top1_idx = tl.argmax(probs, axis=0)
    probs_masked = tl.where(cols == top1_idx, float("-inf"), probs)
    top2_val = tl.max(probs_masked, axis=0)
    top2_idx = tl.argmax(probs_masked, axis=0)
    total = top1_val + top2_val
    tl.store(weight0_ptr + row, top1_val / total)
    tl.store(weight1_ptr + row, top2_val / total)
    tl.store(idx0_ptr + row, top1_idx)
    tl.store(idx1_ptr + row, top2_idx)


def section_21_3():
    print()
    print("=== Section 21.3: top-2 MoE gating vs. torch.topk, and the renormalization trap ===")
    torch.manual_seed(0)
    M, E = 6, 8
    logits = torch.randn(M, E)
    BLOCK_E = triton.next_power_of_2(E)

    probs = torch.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, 2, dim=-1)
    oracle_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    w0 = torch.empty(M)
    w1 = torch.empty(M)
    idx0 = torch.empty(M, dtype=torch.int32)
    idx1 = torch.empty(M, dtype=torch.int32)
    moe_top2_gating_kernel[(M,)](logits, w0, w1, idx0, idx1, M, E, logits.stride(0), BLOCK_E=BLOCK_E)

    indices_match = (idx0.long() == topk_idx[:, 0]).all().item() and (idx1.long() == topk_idx[:, 1]).all().item()
    print("indices match torch.topk?", indices_match)
    print("weights allclose to oracle?",
          torch.allclose(w0, oracle_weights[:, 0], atol=1e-5) and torch.allclose(w1, oracle_weights[:, 1], atol=1e-5))
    print("w0+w1 == 1 for all rows?", torch.allclose(w0 + w1, torch.ones(M)))

    raw_sums = topk_vals.sum(dim=-1)
    print("mean raw (non-renormalized) top-2 weight sum:", raw_sums.mean().item())


if __name__ == "__main__":
    section_21_1_and_2()
    section_21_3()
