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


def section_18_1():
    print("=== Section 18.1: constructing and inspecting triton.Config objects ===")
    cfg = triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3)
    print("kwargs:", cfg.kwargs)
    print("num_warps:", cfg.num_warps)
    print("num_stages:", cfg.num_stages)

    print()
    print("=== Section 18.1: an illegal candidate -- BLOCK=300 is not a power of 2 ===")
    torch.manual_seed(0)
    n = 4096
    x = torch.randn(n)
    try:
        out_bad = torch.empty_like(x)
        gelu_forward_kernel[(triton.cdiv(n, 300),)](x, out_bad, n, BLOCK=300, num_warps=4, num_stages=2)
        print("ran without error (unexpected)")
    except Exception as e:
        print(f"caught, as expected: {type(e).__name__}: {e}")


if __name__ == "__main__":
    section_18_1()
