import torch
import triton
import triton.language as tl

SEG = 32
WARP = 32
FLOAT = 4
FIELDS = ["x", "y", "z", "vx", "vy", "vz", "mass"]
NEEDED = ["vx", "vy", "vz", "mass"]


def segments_touched(addresses):
    return sorted(set(a // SEG for a in addresses))


@triton.jit
def copy_2d_hardcoded_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    ptrs = x_ptr + rows[:, None] * N + cols[None, :]
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


def main():
    print("=== Section 3.1: AoS vs. SoA, warp-wide field reads ===")
    row_stride = len(FIELDS) * FLOAT
    total_segs_aos = set()
    for f in NEEDED:
        col = FIELDS.index(f)
        addrs = [i * row_stride + col * FLOAT for i in range(WARP)]
        total_segs_aos.update(segments_touched(addrs))
    needed_bytes = WARP * len(NEEDED) * FLOAT
    aos_bytes = len(total_segs_aos) * SEG
    print(f"AoS total: {len(total_segs_aos)} distinct 32B segments = {aos_bytes} bytes fetched")
    print(f"needed: {needed_bytes} bytes -> utilization = {needed_bytes/aos_bytes*100:.1f}%")

    total_segs_soa = set()
    for fi, f in enumerate(NEEDED):
        addrs = [i * FLOAT for i in range(WARP)]
        total_segs_soa.update((fi, s) for s in segments_touched(addrs))
    soa_bytes = len(total_segs_soa) * SEG
    print(f"SoA total: {len(total_segs_soa)} distinct 32B segments = {soa_bytes} bytes fetched")
    print(f"needed: {needed_bytes} bytes -> utilization = {needed_bytes/soa_bytes*100:.1f}%")

    print("\n=== Section 3.2: one row vs. one column of a 32x32 row-major matrix ===")
    M, N = 32, 32
    rs, cs = N * 4, 4
    row_addrs = [c * cs for c in range(N)]
    col_addrs = [r * rs for r in range(M)]
    row_segs = segments_touched(row_addrs)
    col_segs = segments_touched(col_addrs)
    print(f"row: {len(row_segs)} segments -> {N*4/(len(row_segs)*SEG)*100:.1f}% utilization")
    print(f"column: {len(col_segs)} segments -> {M*4/(len(col_segs)*SEG)*100:.1f}% utilization")

    print("\n=== Section 3.3: hardcoded strides vs. real strides ===")
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out_contig = torch.empty(3, 4)
    copy_2d_hardcoded_kernel[(1,)](base, out_contig, 3, 4, BLOCK_M=4, BLOCK_N=4)
    print("hardcoded kernel on contiguous base, correct?", torch.equal(out_contig, base))

    t = base.t()
    out_hardcoded = torch.empty(4, 3)
    copy_2d_hardcoded_kernel[(1,)](t, out_hardcoded, 4, 3, BLOCK_M=4, BLOCK_N=4)
    print("hardcoded kernel on base.t(), correct?", torch.equal(out_hardcoded, t))

    out_strided = torch.empty(4, 3)
    copy_2d_strided_kernel[(1,)](t, out_strided, 4, 3, t.stride(0), t.stride(1), BLOCK_M=4, BLOCK_N=4)
    print("stride-aware kernel on base.t(), correct?", torch.equal(out_strided, t))

    print("\n=== Section 3.4: a 2x2 tile of base.t() ===")
    out_tile = torch.empty(2, 2)
    copy_2d_strided_kernel[(1,)](t, out_tile, 2, 2, t.stride(0), t.stride(1), BLOCK_M=2, BLOCK_N=2)
    print("top-left 2x2 tile:", out_tile.tolist(), "(expected [[0,4],[1,5]])")


if __name__ == "__main__":
    main()
