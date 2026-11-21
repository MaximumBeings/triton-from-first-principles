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


def section_19_2():
    print("=== Section 19.2: a genuine, repeatable warmup effect ===")
    torch.manual_seed(0)
    n = 65536
    x = torch.randn(n)
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        gelu_forward_kernel[grid](x, out, n, BLOCK=BLOCK)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print("first 5 calls (seconds):", [f"{t:.6f}" for t in times])
    print("first call vs. later calls -- first is measurably slower:",
          times[0] > 1.5 * sum(times[1:]) / len(times[1:]))

    print()
    print("=== Section 19.2: median vs. mean with a synthetic labeled outlier ===")
    # Genuinely measure 20 trials at a different configuration:
    BLOCK2 = 256
    grid2 = (triton.cdiv(n, BLOCK2),)
    for _ in range(3):  # warmup, discarded
        gelu_forward_kernel[grid2](x, out, n, BLOCK=BLOCK2)
    real_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        gelu_forward_kernel[grid2](x, out, n, BLOCK=BLOCK2)
        t1 = time.perf_counter()
        real_times.append(t1 - t0)
    real_times.sort()
    median_real = real_times[len(real_times) // 2]
    mean_real = sum(real_times) / len(real_times)
    print(f"20 real trials: median={median_real*1000:.2f}ms  mean={mean_real*1000:.2f}ms")

    # Purely illustrative: one synthetic 3.0s outlier added to the real data,
    # standing in for a real-world scheduling hiccup -- NOT a genuine measurement.
    with_outlier = real_times + [3.0]
    with_outlier.sort()
    median_outlier = with_outlier[len(with_outlier) // 2]
    mean_outlier = sum(with_outlier) / len(with_outlier)
    print(f"same 20 + 1 synthetic 3.0s outlier: median={median_outlier*1000:.2f}ms  mean={mean_outlier*1000:.2f}ms")
    print(f"median shifted by {(median_outlier-median_real)*1000:.3f}ms; "
          f"mean shifted by {(mean_outlier-mean_real)*1000:.2f}ms")


if __name__ == "__main__":
    section_19_2()
