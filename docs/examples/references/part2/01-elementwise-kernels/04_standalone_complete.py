import torch
import triton
import triton.language as tl


@triton.jit
def fma_relu_kernel(x_ptr, a, b, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = a * x + b
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


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


def fused_fma_relu(x, a, b):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    fma_relu_kernel[(1,)](x, a, b, out, n, BLOCK=BLOCK)
    return out


def triton_gelu(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    gelu_exact_kernel[(1,)](x, out, n, BLOCK=BLOCK)
    return out


def triton_gelu_tanh(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    gelu_tanh_approx_kernel[(1,)](x, out, n, BLOCK=BLOCK)
    return out


def main():
    print("=== Section 12.1: complete fused kernel ===")
    xf = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    a, b = 2.0, -1.0
    out = fused_fma_relu(xf, a, b)
    ref = torch.relu(a * xf + b)
    print("fused result:", out.tolist(), " match?", torch.equal(out, ref))

    print("\n=== Section 12.2: fused vs. naive chain, identical result ===")
    y3 = torch.relu((a * xf) + b)
    print("naive chain:", y3.tolist(), " identical to fused?", torch.equal(y3, out))

    print("\n=== Section 12.3: GELU vs. oracle ===")
    x = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    gelu_out = triton_gelu(x)
    oracle = torch.nn.functional.gelu(x, approximate="none")
    print("exact GELU allclose to oracle?", torch.allclose(gelu_out, oracle, atol=1e-6))

    tanh_out = triton_gelu_tanh(x)
    oracle_tanh = torch.nn.functional.gelu(x, approximate="tanh")
    print("tanh-approx allclose to its own oracle?", torch.allclose(tanh_out, oracle_tanh, atol=1e-6))
    print("tanh-approx max diff vs EXACT oracle:", (tanh_out - oracle).abs().max().item())


if __name__ == "__main__":
    main()
