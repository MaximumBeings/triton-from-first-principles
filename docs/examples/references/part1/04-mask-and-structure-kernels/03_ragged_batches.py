import torch
import triton
import triton.language as tl


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


def demo_ragged_batches():
    print("=== Section 9.3: a per-row bound instead of a shared n ===")
    padded = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 999.0, 999.0],
        [5.0, 5.0, 5.0, 999.0],
    ])
    lengths = torch.tensor([4, 2, 3], dtype=torch.int32)

    out = torch.empty(3, dtype=torch.float32)
    ragged_row_sum_kernel[(3,)](padded, lengths, out, padded.stride(0), MAX_WIDTH=4)

    print("padded tensor:\n", padded)
    print("lengths:", lengths.tolist())
    print("ragged row sums (ignoring padding):", out.tolist())


if __name__ == "__main__":
    demo_ragged_batches()
