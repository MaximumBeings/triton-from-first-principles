import torch
import triton
import triton.language as tl


@triton.jit
def broadcast_add_kernel(out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    combined = rows[:, None] + cols[None, :]
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
    tl.store(out_ptr + tl.arange(0, BLOCK), combined)


@triton.jit
def hinted_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    offsets = tl.multiple_of(offsets, BLOCK)
    offsets = tl.max_contiguous(offsets, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


@triton.jit
def copy_block_ptr_kernel(x_ptr, out_ptr, M, N, stride_m, stride_n,
                           out_M, out_N,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M, N), strides=(stride_m, stride_n),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tile = tl.load(x_block_ptr, boundary_check=(0, 1))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(out_M, out_N), strides=(out_N, 1),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(out_block_ptr, tile, boundary_check=(0, 1))


def main():
    print("=== Section 7.1: broadcasting ===")
    out = torch.empty(3, 4, dtype=torch.int32)
    broadcast_add_kernel[(1,)](out, 3, 4, BLOCK_M=4, BLOCK_N=4)
    print("broadcast result (row+col):\n", out)

    out2 = torch.empty(4, dtype=torch.int32)
    try:
        bad_broadcast_kernel[(1,)](out2, BLOCK_A=4, BLOCK_B=8)
        print("unexpectedly succeeded:", out2.tolist())
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

    out3 = torch.empty(4, dtype=torch.int32)
    forgot_none_kernel[(1,)](out3, BLOCK=4)
    print("forgot-None result:", out3.tolist())

    print("\n=== Section 7.2: alignment hints ===")
    n = 16
    x = torch.arange(n, dtype=torch.float32)
    out4 = torch.empty(n)
    BLOCK = 8
    grid = (triton.cdiv(n, BLOCK),)
    hinted_kernel[grid](x, out4, n, BLOCK=BLOCK)
    print("hinted kernel result matches x*2?", torch.equal(out4, x * 2))

    print("\n=== Section 7.3: block pointers ===")
    real = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out_correct = torch.empty(3, 4)
    copy_block_ptr_kernel[(1,)](real, out_correct, 3, 4, real.stride(0), real.stride(1), 3, 4, BLOCK_M=4, BLOCK_N=4)
    print("correct block_ptr shape/strides -> matches?", torch.equal(out_correct, real))

    out_lied = torch.empty(4, 3)
    copy_block_ptr_kernel[(1,)](real, out_lied, 4, 3, 3, 1, 4, 3, BLOCK_M=4, BLOCK_N=4)
    print("LIED shape (4,3) via block_ptr, result:\n", out_lied)


if __name__ == "__main__":
    main()
