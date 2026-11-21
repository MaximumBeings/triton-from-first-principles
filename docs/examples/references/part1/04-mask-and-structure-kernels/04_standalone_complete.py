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
def block_sparse_double_kernel(x_ptr, out_ptr, block_mask_ptr, N,
                                BLOCK: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    is_active = tl.load(block_mask_ptr + pid)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    if is_active:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x * 2.0, mask=mask)


@triton.jit
def ragged_row_sum_kernel(x_ptr, lengths_ptr, out_ptr, stride_row, MAX_WIDTH: tl.constexpr):
    row = tl.program_id(0)
    row_len = tl.load(lengths_ptr + row)
    cols = tl.arange(0, MAX_WIDTH)
    mask = cols < row_len
    ptrs = x_ptr + row * stride_row + cols
    x = tl.load(ptrs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    tl.store(out_ptr + row, total)


def main():
    print("=== Section 9.1: triangular masks ===")
    x = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    out = torch.empty(4, 4)
    causal_mask_kernel[(1,)](x, out, 4, 4, x.stride(0), x.stride(1), BLOCK_M=4, BLOCK_N=4)
    print("causal-masked:\n", out)
    print("matches torch.tril(x)?", torch.equal(out, torch.tril(x)))

    print("\n=== Section 9.2: block-sparse skipping ===")
    N, BLOCK = 16, 4
    NUM_BLOCKS = triton.cdiv(N, BLOCK)
    xb = torch.arange(N, dtype=torch.float32)
    outb = torch.zeros(N)
    block_mask = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    block_sparse_double_kernel[(NUM_BLOCKS,)](xb, outb, block_mask, N, BLOCK=BLOCK, NUM_BLOCKS=NUM_BLOCKS)
    print("block-sparse result:", outb.tolist())

    print("\n=== Section 9.3: ragged batches ===")
    padded = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 999.0, 999.0],
        [5.0, 5.0, 5.0, 999.0],
    ])
    lengths = torch.tensor([4, 2, 3], dtype=torch.int32)
    outr = torch.empty(3, dtype=torch.float32)
    ragged_row_sum_kernel[(3,)](padded, lengths, outr, padded.stride(0), MAX_WIDTH=4)
    print("ragged row sums:", outr.tolist())


if __name__ == "__main__":
    main()
