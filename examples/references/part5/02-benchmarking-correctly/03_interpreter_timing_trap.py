import torch
import triton
import triton.language as tl
import time


@triton.jit
def gelu_forward_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    cdf = 0.5 * (1.0 + tl.math.erf(x * SQRT_2_INV))
    out = x * cdf
    tl.store(out_ptr + offsets, out, mask=mask)


def timed_median(BLOCK, n, x, out, n_trials=20, n_warmup=3):
    grid = (triton.cdiv(n, BLOCK),)
    for _ in range(n_warmup):
        gelu_forward_kernel[grid](x, out, n, BLOCK=BLOCK)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        gelu_forward_kernel[grid](x, out, n, BLOCK=BLOCK)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    return times[len(times) // 2], min(times), max(times)


def section_19_3():
    print("=== Section 19.3: BLOCK=256 (256 programs) vs. BLOCK=4096 (16 programs) ===")
    torch.manual_seed(0)
    n = 65536
    x = torch.randn(n)
    out = torch.empty_like(x)

    median_256, min_256, max_256 = timed_median(256, n, x, out)
    median_4096, min_4096, max_4096 = timed_median(4096, n, x, out)

    print(f"BLOCK=  256  n_programs={triton.cdiv(n, 256):4d}  "
          f"median={median_256*1000:.4f}ms  min={min_256*1000:.4f}ms  max={max_256*1000:.4f}ms")
    print(f"BLOCK= 4096  n_programs={triton.cdiv(n, 4096):4d}  "
          f"median={median_4096*1000:.4f}ms  min={min_4096*1000:.4f}ms  max={max_4096*1000:.4f}ms")
    print(f"ratio (BLOCK=256 median / BLOCK=4096 median): {median_256/median_4096:.2f}x")
    print("This ratio reflects Python-level per-program interpreter overhead,")
    print("NOT a real GPU's occupancy/parallelism trade-off -- see Section 19.3's COMMON TRAP.")


if __name__ == "__main__":
    section_19_3()
