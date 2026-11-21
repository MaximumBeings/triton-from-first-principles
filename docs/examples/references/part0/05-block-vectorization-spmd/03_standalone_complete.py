import torch
import triton
import triton.language as tl


@triton.jit
def double_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


@triton.jit
def double_kernel_grid(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


@triton.jit
def sum_gridstride_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, NUM_CHUNKS: tl.constexpr):
    offsets0 = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in range(NUM_CHUNKS):
        offsets = c * BLOCK + offsets0
        mask = offsets < n
        acc += tl.load(x_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr, total)


@triton.jit
def transpose_tile_kernel(x_ptr, out_ptr, M, N,
                           stride_xm, stride_xn, stride_om, stride_on,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x_ptrs = x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn
    tile = tl.load(x_ptrs, mask=mask, other=0.0)
    out_ptrs = out_ptr + cols[None, :] * stride_om + rows[:, None] * stride_on
    tl.store(out_ptrs, tile, mask=mask)


def main():
    print("=== Section 5.1: BLOCK must be a power of 2 ===")
    x7 = torch.arange(7, dtype=torch.float32)
    out7 = torch.empty(7)
    try:
        double_kernel[(1,)](x7, out7, 7, BLOCK=7)
        print("BLOCK=7 succeeded:", out7.tolist())
    except Exception as e:
        print(f"BLOCK=7 -> {type(e).__name__}: {e}")
    out7b = torch.empty(7)
    double_kernel[(1,)](x7, out7b, 7, BLOCK=8)
    print("BLOCK=8 (next power of 2, masked down to n=7):", out7b.tolist())

    print("\n=== Section 5.2: one masked kernel handles the 'remainder' automatically ===")
    n, BLOCK = 10, 8
    grid = (triton.cdiv(n, BLOCK),)
    x10 = torch.arange(10, dtype=torch.float32)
    out10 = torch.empty(10)
    double_kernel_grid[grid](x10, out10, n, BLOCK=BLOCK)
    print(f"n={n}, BLOCK={BLOCK}, grid={grid} -> one kernel, no separate remainder kernel")
    print("result:", out10.tolist(), "correct?", torch.equal(out10, x10 * 2))

    print("\n=== Section 5.3: accumulate wide, reduce narrow ===")
    N, BLOCK = 1000, 128
    NUM_CHUNKS = triton.cdiv(N, BLOCK)
    xN = torch.arange(N, dtype=torch.float32)
    out_sum = torch.empty(1, dtype=torch.float32)
    sum_gridstride_kernel[(1,)](xN, out_sum, N, BLOCK=BLOCK, NUM_CHUNKS=NUM_CHUNKS)
    print(f"N={N}, BLOCK={BLOCK}, NUM_CHUNKS={NUM_CHUNKS}")
    print("triton sum:", out_sum.item(), "torch sum:", xN.sum().item())

    print("\n=== Section 5.4: SPMD block transpose, one program per tile ===")
    M, N2 = 4, 6
    BLOCK_M, BLOCK_N = 2, 2
    A = torch.arange(M * N2, dtype=torch.float32).reshape(M, N2)
    out_t = torch.empty(N2, M, dtype=torch.float32)
    grid_t = (triton.cdiv(M, BLOCK_M), triton.cdiv(N2, BLOCK_N))
    print("grid (programs):", grid_t, "=", grid_t[0] * grid_t[1], "programs total")
    transpose_tile_kernel[grid_t](
        A, out_t, M, N2,
        A.stride(0), A.stride(1), out_t.stride(0), out_t.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    print("matches A.t()?", torch.equal(out_t, A.t()))


if __name__ == "__main__":
    main()
