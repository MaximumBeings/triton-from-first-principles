import torch
import triton
import triton.language as tl


@triton.jit
def copy_2d_hardcoded_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    ptrs = x_ptr + rows[:, None] * N + cols[None, :]   # assumes row-major, stride_row=N
    x = tl.load(ptrs, mask=mask)
    out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
    tl.store(out_ptrs, x, mask=mask)


@triton.jit
def copy_2d_strided_kernel(x_ptr, out_ptr, M, N, stride_row, stride_col,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    ptrs = x_ptr + rows[:, None] * stride_row + cols[None, :] * stride_col
    x = tl.load(ptrs, mask=mask)
    out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
    tl.store(out_ptrs, x, mask=mask)


def demo_transpose_trap():
    print("=== Section 3.3: hardcoded strides vs. real strides ===")
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    out_contig = torch.empty(3, 4)
    copy_2d_hardcoded_kernel[(1,)](base, out_contig, 3, 4, BLOCK_M=4, BLOCK_N=4)
    print("hardcoded kernel on contiguous base, correct?", torch.equal(out_contig, base))

    t = base.t()
    print("base.t() shape:", t.shape, "stride:", t.stride(), "is_contiguous:", t.is_contiguous())

    out_hardcoded = torch.empty(4, 3)
    copy_2d_hardcoded_kernel[(1,)](t, out_hardcoded, 4, 3, BLOCK_M=4, BLOCK_N=4)
    print("hardcoded kernel on base.t(), correct?", torch.equal(out_hardcoded, t))

    out_strided = torch.empty(4, 3)
    copy_2d_strided_kernel[(1,)](t, out_strided, 4, 3, t.stride(0), t.stride(1), BLOCK_M=4, BLOCK_N=4)
    print("stride-aware kernel on base.t(), correct?", torch.equal(out_strided, t))
    return base, t


def demo_2d_tile(t):
    print("\n=== Section 3.4: a 2x2 tile of base.t(), read via the stride-aware kernel ===")
    out = torch.empty(2, 2)
    copy_2d_strided_kernel[(1,)](t, out, 2, 2, t.stride(0), t.stride(1), BLOCK_M=2, BLOCK_N=2)
    print("top-left 2x2 tile:", out.tolist(), "(expected [[0,4],[1,5]])")


if __name__ == "__main__":
    _, t = demo_transpose_trap()
    demo_2d_tile(t)
