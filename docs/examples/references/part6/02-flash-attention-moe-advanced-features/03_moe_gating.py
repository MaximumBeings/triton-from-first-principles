import torch
import triton
import triton.language as tl


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


def moe_top2_gating(logits):
    M, E = logits.shape
    BLOCK_E = triton.next_power_of_2(E)
    w0 = torch.empty(M)
    w1 = torch.empty(M)
    idx0 = torch.empty(M, dtype=torch.int32)
    idx1 = torch.empty(M, dtype=torch.int32)
    moe_top2_gating_kernel[(M,)](logits, w0, w1, idx0, idx1, M, E, logits.stride(0), BLOCK_E=BLOCK_E)
    return w0, w1, idx0, idx1


def section_21_3():
    print("=== Section 21.3: top-2 MoE gating vs. torch.topk ===")
    torch.manual_seed(0)
    M, E = 6, 8
    logits = torch.randn(M, E)

    probs = torch.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, 2, dim=-1)
    oracle_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    w0, w1, idx0, idx1 = moe_top2_gating(logits)
    indices_match = (idx0.long() == topk_idx[:, 0]).all().item() and (idx1.long() == topk_idx[:, 1]).all().item()
    print("indices match torch.topk?", indices_match)
    print("weights allclose to oracle?",
          torch.allclose(w0, oracle_weights[:, 0], atol=1e-5) and torch.allclose(w1, oracle_weights[:, 1], atol=1e-5))
    print("w0+w1 == 1 for all rows?", torch.allclose(w0 + w1, torch.ones(M)))

    print()
    print("=== Section 21.3: forgetting to renormalize ===")
    raw_sums = topk_vals.sum(dim=-1)
    print("raw (non-renormalized) top-2 weight sums per row:", raw_sums)
    print("mean raw sum (should be 1.0 if renormalized; isn't, here):", raw_sums.mean().item())


if __name__ == "__main__":
    section_21_3()
