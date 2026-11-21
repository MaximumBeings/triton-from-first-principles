import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_max_kernel(x_ptr, sum_ptr, max_ptr, stride_row, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    row_sum = tl.sum(x, axis=0)
    x_for_max = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float('-inf'))
    row_max = tl.max(x_for_max, axis=0)
    tl.store(sum_ptr + row, row_sum)
    tl.store(max_ptr + row, row_max)


def demo_row_reductions():
    print("=== Section 14.1: explicit row sum and row max ===")
    M, N = 4, 6
    x = torch.tensor([
        [1.0, 5.0, 2.0, 0.0, 3.0, -1.0],
        [-2.0, -1.0, -3.0, -4.0, 0.0, 1.0],
        [10.0, 20.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    sums = torch.empty(M)
    maxes = torch.empty(M)
    row_sum_max_kernel[(M,)](x, sums, maxes, x.stride(0), N, BLOCK_N=8)
    print("row sums:", sums.tolist())
    print("row maxes:", maxes.tolist())
    print("sums match?", torch.equal(sums, x.sum(dim=1)))
    print("maxes match?", torch.equal(maxes, x.max(dim=1).values))


if __name__ == "__main__":
    demo_row_reductions()
