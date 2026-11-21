import torch
import triton
import triton.language as tl


@triton.jit
def copy_2d_strided_kernel(x_ptr, out_ptr, M, N, stride_row, stride_col,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    ptrs = x_ptr + rows[:, None] * stride_row + cols[None, :] * stride_col
    v = tl.load(ptrs, mask=mask)
    out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
    tl.store(out_ptrs, v, mask=mask)


def demo_shape_stride_promise():
    print("=== Section 6.2: lying about the shape/stride arguments ===")
    real = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("real tensor (3x4):\n", real)

    out_correct = torch.empty(3, 4)
    copy_2d_strided_kernel[(1,)](real, out_correct, 3, 4, real.stride(0), real.stride(1), BLOCK_M=4, BLOCK_N=4)
    print("correct M,N,strides -> matches?", torch.equal(out_correct, real))

    out_lied = torch.empty(4, 3)
    copy_2d_strided_kernel[(1,)](real, out_lied, 4, 3, 3, 1, BLOCK_M=4, BLOCK_N=4)
    print("LIED shape (4,3) on a real (3,4) tensor, result:\n", out_lied)
    print("(no error raised anywhere -- just a differently-shaped reinterpretation of the same bytes)")


if __name__ == "__main__":
    demo_shape_stride_promise()
