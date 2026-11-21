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


@triton.jit
def square_forward_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * x, mask=mask)


@triton.jit
def square_backward_kernel(grad_out_ptr, x_ptr, grad_in_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    tl.store(grad_in_ptr + offsets, grad_out * 2.0 * x, mask=mask)


def _launch(fwd_kernel, x, BLOCK=1024):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    fwd_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out


def _launch_bwd(bwd_kernel, grad_output, x, BLOCK=1024):
    grad_input = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    bwd_kernel[grid](grad_output, x, grad_input, n, BLOCK=BLOCK)
    return grad_input


order = []


class TritonGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = _launch(gelu_forward_kernel, x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        order.append("TritonGELU.backward")
        x, = ctx.saved_tensors
        return _launch_bwd(gelu_backward_kernel, grad_output.contiguous(), x)


class TritonSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = _launch(square_forward_kernel, x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        order.append("TritonSquare.backward")
        x, = ctx.saved_tensors
        return _launch_bwd(square_backward_kernel, grad_output.contiguous(), x)


def section_17_2():
    print("=== Section 17.2: TritonGELU then TritonSquare -- graph structure and call order ===")
    torch.manual_seed(4)
    x = torch.randn(4, requires_grad=True)
    z = TritonGELU.apply(x)
    y = TritonSquare.apply(z)
    loss = y.sum()

    print("y.grad_fn:", y.grad_fn)
    print("y.grad_fn.next_functions:", y.grad_fn.next_functions)
    print("z.grad_fn:", z.grad_fn)
    print("z.grad_fn.next_functions:", z.grad_fn.next_functions)

    order.clear()
    loss.backward()
    print("backward call order:", order)

    x_oracle = x.detach().clone().requires_grad_(True)
    z_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    (z_oracle * z_oracle).sum().backward()
    print("x.grad allclose to oracle?", torch.allclose(x.grad, x_oracle.grad, atol=1e-5))


if __name__ == "__main__":
    section_17_2()
