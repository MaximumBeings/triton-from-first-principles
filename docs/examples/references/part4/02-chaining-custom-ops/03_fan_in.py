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


captured = []


class TritonGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = triton_gelu_forward(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        captured.append(grad_output.clone())
        x, = ctx.saved_tensors
        return triton_gelu_backward(grad_output.contiguous(), x)


def section_17_3():
    print("=== Section 17.3: z used once vs. z used twice (same op) ===")
    torch.manual_seed(5)
    x1 = torch.randn(4, requires_grad=True)

    captured.clear()
    z1 = TritonGELU.apply(x1)
    z1.sum().backward()
    print("used once  -> grad_output:", captured[-1])

    captured.clear()
    x2 = x1.detach().clone().requires_grad_(True)
    z2 = TritonGELU.apply(x2)
    (z2.sum() + z2.sum()).backward()
    print("used twice -> grad_output:", captured[-1])
    print("number of times backward was called:", len(captured))

    x_oracle = x1.detach().clone().requires_grad_(True)
    z_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    (z_oracle.sum() + z_oracle.sum()).backward()
    print("x2.grad allclose to oracle?", torch.allclose(x2.grad, x_oracle.grad, atol=1e-5))

    print()
    print("=== Section 17.3: z used in two DIFFERENT downstream expressions ===")
    captured.clear()
    x3 = x1.detach().clone().requires_grad_(True)
    z3 = TritonGELU.apply(x3)
    ((2 * z3).sum() + (3 * z3).sum()).backward()
    print("grad_output (expect uniformly 5.0):", captured[-1])

    x_oracle2 = x1.detach().clone().requires_grad_(True)
    z_oracle2 = torch.nn.functional.gelu(x_oracle2, approximate="none")
    ((2 * z_oracle2).sum() + (3 * z_oracle2).sum()).backward()
    print("x3.grad allclose to oracle?", torch.allclose(x3.grad, x_oracle2.grad, atol=1e-5))


if __name__ == "__main__":
    section_17_3()
