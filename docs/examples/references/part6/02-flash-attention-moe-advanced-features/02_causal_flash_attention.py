import torch
import triton
import triton.language as tl
import math


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


def _run(kernel, q, k, v, BLOCK_M=16, BLOCK_N=16):
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


def section_21_2():
    print("=== Section 21.2: causal Flash Attention vs. naive causal attention ===")
    torch.manual_seed(0)
    M = N = 20
    D = 8
    q = torch.randn(M, D)
    k = torch.randn(N, D)
    v = torch.randn(N, D)
    scale = 1.0 / math.sqrt(D)

    scores_full = q @ k.T * scale
    causal_mask = torch.tril(torch.ones(M, N, dtype=torch.bool))
    oracle_causal = torch.softmax(scores_full.masked_fill(~causal_mask, float("-inf")), dim=-1) @ v
    oracle_noncausal = torch.softmax(scores_full, dim=-1) @ v

    out_causal = _run(flash_attn_causal_kernel, q, k, v)
    print("causal kernel allclose to causal oracle?", torch.allclose(out_causal, oracle_causal, atol=1e-4))

    print()
    print("=== Section 21.2: a correct (non-causal) kernel applied to the wrong task ===")
    out_noncausal = _run(flash_attn_kernel, q, k, v)
    print("non-causal kernel allclose to CAUSAL oracle?", torch.allclose(out_noncausal, oracle_causal, atol=1e-4))
    print("non-causal kernel allclose to NON-causal oracle?",
          torch.allclose(out_noncausal, oracle_noncausal, atol=1e-4))


if __name__ == "__main__":
    section_21_2()
