import torch
import triton
import triton.language as tl


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
        print("ran without error (unexpected in this environment)")
    except RuntimeError as e:
        print(f"caught, as expected: {type(e).__name__}: {e}")
        print("(same root cause as Chapter 18's @triton.autotune failure: no GPU driver)")


if __name__ == "__main__":
    section_19_1()
