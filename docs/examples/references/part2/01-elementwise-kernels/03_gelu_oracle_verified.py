import torch
import triton
import triton.language as tl


@triton.jit
def gelu_exact_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    cdf = 0.5 * (1.0 + tl.math.erf(x * SQRT_2_INV))
    y = x * cdf
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def gelu_tanh_approx_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    COEFF: tl.constexpr = 0.044715
    inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    cdf = 0.5 * (1.0 + tanh_inner)
    y = x * cdf
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    gelu_exact_kernel[(1,)](x, out, n, BLOCK=BLOCK)
    return out


def triton_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    gelu_tanh_approx_kernel[(1,)](x, out, n, BLOCK=BLOCK)
    return out


def demo_gelu_oracle():
    print("=== Section 12.3: exact GELU vs. its independent oracle ===")
    x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    out = triton_gelu(x)
    oracle = torch.nn.functional.gelu(x, approximate="none")
    print("triton GELU:  ", out.tolist())
    print("torch oracle: ", oracle.tolist())
    print("max abs diff:", (out - oracle).abs().max().item())
    print("allclose?", torch.allclose(out, oracle, atol=1e-6))

    print("\n=== tanh-approximation GELU: matches its own oracle, differs from the exact one ===")
    out_tanh = triton_gelu_tanh(x)
    oracle_tanh = torch.nn.functional.gelu(x, approximate="tanh")
    print("triton tanh-approx GELU: ", out_tanh.tolist())
    print("torch tanh-approx oracle:", oracle_tanh.tolist())
    print("max abs diff vs tanh oracle:", (out_tanh - oracle_tanh).abs().max().item())
    print("max abs diff vs EXACT gelu oracle:", (out_tanh - oracle).abs().max().item())


if __name__ == "__main__":
    demo_gelu_oracle()
