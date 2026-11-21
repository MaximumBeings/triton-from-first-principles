import torch
import triton
import triton.language as tl


@triton.jit
def broadcast_add_kernel(out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_col = rows[:, None]
    col_row = cols[None, :]
    tl.static_print("row_col shape:", row_col.shape)
    tl.static_print("col_row shape:", col_row.shape)
    combined = row_col + col_row
    tl.static_print("combined shape:", combined.shape)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
    tl.store(out_ptrs, combined, mask=mask)


@triton.jit
def bad_broadcast_kernel(out_ptr, BLOCK_A: tl.constexpr, BLOCK_B: tl.constexpr):
    a = tl.arange(0, BLOCK_A)
    b = tl.arange(0, BLOCK_B)
    c = a + b
    tl.store(out_ptr + tl.arange(0, BLOCK_A), c)


@triton.jit
def forgot_none_kernel(out_ptr, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    combined = rows + cols
    tl.static_print("combined shape (forgot None-indexing):", combined.shape)
    tl.store(out_ptr + tl.arange(0, BLOCK), combined)


def demo_broadcasting():
    print("=== Section 7.1: broadcasting a (BLOCK_M,1) column against a (1,BLOCK_N) row ===")
    out = torch.empty(3, 4, dtype=torch.int32)
    broadcast_add_kernel[(1,)](out, 3, 4, BLOCK_M=4, BLOCK_N=4)
    print("broadcast result (row+col):\n", out)

    print("\n=== incompatible shapes, rejected at trace time ===")
    out2 = torch.empty(4, dtype=torch.int32)
    try:
        bad_broadcast_kernel[(1,)](out2, BLOCK_A=4, BLOCK_B=8)
        print("unexpectedly succeeded:", out2.tolist())
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

    print("\n=== the trap: forgetting None-indexing on equal-length ranges ===")
    out3 = torch.empty(4, dtype=torch.int32)
    forgot_none_kernel[(1,)](out3, BLOCK=4)
    print("forgot-None result (1D, NOT the 4x4 outer grid you probably wanted):", out3.tolist())


if __name__ == "__main__":
    demo_broadcasting()
