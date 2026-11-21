import torch
import triton
import triton.language as tl


@triton.jit
def causal_mask_kernel(x_ptr, out_ptr, M, N, stride_m, stride_n,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    bounds_mask = (rows[:, None] < M) & (cols[None, :] < N)
    causal_mask = cols[None, :] <= rows[:, None]
    ptrs = x_ptr + rows[:, None] * stride_m + cols[None, :] * stride_n
    x = tl.load(ptrs, mask=bounds_mask, other=0.0)
    result = tl.where(causal_mask, x, 0.0)
    out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
    tl.store(out_ptrs, result, mask=bounds_mask)


@triton.jit
def wrong_causal_mask_kernel(x_ptr, out_ptr, M, N, stride_m, stride_n,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    bounds_mask = (rows[:, None] < M) & (cols[None, :] < N)
    wrong_mask = cols[None, :] < rows[:, None]     # BUG: should be <=
    ptrs = x_ptr + rows[:, None] * stride_m + cols[None, :] * stride_n
    x = tl.load(ptrs, mask=bounds_mask, other=0.0)
    result = tl.where(wrong_mask, x, 0.0)
    out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
    tl.store(out_ptrs, result, mask=bounds_mask)


def demo_causal_mask():
    print("=== Section 9.1: a correct causal (lower-triangular) mask ===")
    x = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    print("input:\n", x)
    out = torch.empty(4, 4)
    causal_mask_kernel[(1,)](x, out, 4, 4, x.stride(0), x.stride(1), BLOCK_M=4, BLOCK_N=4)
    print("causal-masked (lower triangular, incl. diagonal):\n", out)
    print("matches torch.tril(x)?", torch.equal(out, torch.tril(x)))

    print("\n=== the off-by-one trap: < instead of <= excludes the diagonal ===")
    out_wrong = torch.empty(4, 4)
    wrong_causal_mask_kernel[(1,)](x, out_wrong, 4, 4, x.stride(0), x.stride(1), BLOCK_M=4, BLOCK_N=4)
    print("with < instead of <=:\n", out_wrong)


if __name__ == "__main__":
    demo_causal_mask()
