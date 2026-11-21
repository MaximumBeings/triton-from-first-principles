SEG = 32  # bytes per memory transaction segment (simplified)
WARP = 32  # threads in a warp
FLOAT = 4  # bytes per float32
FIELDS = ["x", "y", "z", "vx", "vy", "vz", "mass"]
NEEDED = ["vx", "vy", "vz", "mass"]


def segments_touched(addresses):
    return sorted(set(a // SEG for a in addresses))


def demo_aos_vs_soa():
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
    print(f"\nSoA total: {len(total_segs_soa)} distinct 32B segments = {soa_bytes} bytes fetched")
    print(f"needed: {needed_bytes} bytes -> utilization = {needed_bytes/soa_bytes*100:.1f}%")


def demo_row_vs_column():
    print("\n=== Section 3.2: one row vs. one column of a 32x32 row-major matrix ===")
    M, N = 32, 32
    row_stride, col_stride = N * 4, 4
    row_addrs = [c * col_stride for c in range(N)]
    col_addrs = [r * row_stride for r in range(M)]
    row_segs = segments_touched(row_addrs)
    col_segs = segments_touched(col_addrs)
    print(f"Reading one row: {len(row_segs)} segments, {len(row_segs)*SEG} bytes fetched "
          f"for {N*4} bytes needed -> {N*4/(len(row_segs)*SEG)*100:.1f}%")
    print(f"Reading one column: {len(col_segs)} segments, {len(col_segs)*SEG} bytes fetched "
          f"for {M*4} bytes needed -> {M*4/(len(col_segs)*SEG)*100:.1f}%")


if __name__ == "__main__":
    demo_aos_vs_soa()
    demo_row_vs_column()
