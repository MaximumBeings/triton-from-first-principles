"""
Chapter 19 standalone: the genuine failure of triton.testing.do_bench without
a GPU driver, a from-scratch warmup-and-robust-statistics demonstration using
only time.perf_counter(), and the interpreter-timing-inversion finding that
closes the chapter. Timing numbers will differ from run to run and machine to
machine; the qualitative findings described in the chapter text are what's
reproducible, not the exact millisecond figures.
"""
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


def section_19_1():
    print("=== Section 19.1: attempting triton.testing.do_bench itself ===")
    torch.manual_seed(0)
    n = 4096
    x = torch.randn(n)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)

    def run():
        gelu_forward_kernel[grid](x, out, n, BLOCK=256)

    try:
        triton.testing.do_bench(run)
        print("ran without error (unexpected)")
    except RuntimeError as e:
        print(f"caught, as expected: {type(e).__name__}")


def section_19_2():
    print()
    print("=== Section 19.2: warmup effect and median vs. mean ===")
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
    print("first call slower than steady-state average?", times[0] > 1.3 * (sum(times[1:]) / len(times[1:])))

    BLOCK2 = 256
    grid2 = (triton.cdiv(n, BLOCK2),)
    for _ in range(3):
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

    with_outlier = real_times + [3.0]
    with_outlier.sort()
    median_outlier = with_outlier[len(with_outlier) // 2]
    mean_outlier = sum(with_outlier) / len(with_outlier)
    print(f"median unaffected by synthetic outlier? {abs(median_outlier - median_real) < 1e-9}")
    print(f"mean shifted by synthetic outlier by {(mean_outlier - mean_real)*1000:.1f}ms")


def section_19_3():
    print()
    print("=== Section 19.3: interpreter timing inversion (BLOCK=256 vs. BLOCK=4096) ===")
    torch.manual_seed(0)
    n = 65536
    x = torch.randn(n)
    out = torch.empty_like(x)

    def timed_median(BLOCK):
        grid = (triton.cdiv(n, BLOCK),)
        for _ in range(3):
            gelu_forward_kernel[grid](x, out, n, BLOCK=BLOCK)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            gelu_forward_kernel[grid](x, out, n, BLOCK=BLOCK)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        times.sort()
        return times[len(times) // 2]

    median_256 = timed_median(256)
    median_4096 = timed_median(4096)
    print(f"BLOCK=256 median: {median_256*1000:.2f}ms   BLOCK=4096 median: {median_4096*1000:.2f}ms")
    print(f"BLOCK=256 is slower under this interpreter? {median_256 > median_4096}")
    print("(fewer programs measuring 'faster' here reflects interpreter overhead,")
    print(" not real GPU occupancy -- never extrapolate this to real hardware)")


if __name__ == "__main__":
    section_19_1()
    section_19_2()
    section_19_3()
