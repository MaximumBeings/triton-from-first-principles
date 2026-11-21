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


@triton.jit
def gelu_backward_kernel(grad_out_ptr, x_ptr, grad_in_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    SQRT_2PI_INV: tl.constexpr = 0.3989422804014327
    cdf = 0.5 * (1.0 + tl.math.erf(x * SQRT_2_INV))
    pdf = SQRT_2PI_INV * tl.exp(-0.5 * x * x)
    d_gelu = cdf + x * pdf
    grad_in = grad_out * d_gelu
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


def triton_gelu_forward(x, BLOCK=1024):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    gelu_forward_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out


def triton_gelu_backward(grad_output, x, BLOCK=1024):
    x = x.contiguous()
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    gelu_backward_kernel[grid](grad_output, x, grad_input, n, BLOCK=BLOCK)
    return grad_input


def section_16_2():
    print("=== Section 16.2: masked multi-program launch, n=10000, BLOCK=1024 ===")
    torch.manual_seed(0)
    n = 10000
    BLOCK = 1024
    print(f"triton.cdiv(n, BLOCK) = {triton.cdiv(n, BLOCK)} programs")

    x = torch.randn(n)
    out = triton_gelu_forward(x, BLOCK=BLOCK)
    oracle_out = torch.nn.functional.gelu(x, approximate="none")
    print("forward allclose to oracle?", torch.allclose(out, oracle_out, atol=1e-5))

    grad_output = torch.randn(n)
    grad_input = triton_gelu_backward(grad_output, x, BLOCK=BLOCK)
    x_oracle = x.clone().requires_grad_(True)
    y_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    y_oracle.backward(grad_output)
    print("backward allclose to oracle?", torch.allclose(grad_input, x_oracle.grad, atol=1e-5))


if __name__ == "__main__":
    section_16_2()
