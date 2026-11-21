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


class TritonGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK = triton.next_power_of_2(n)
        gelu_exact_kernel[(1,)](x, out, n, BLOCK=BLOCK)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # placeholder: correct analytical formula in plain PyTorch,
        # not yet a hand-written Triton kernel (Chapter 16).
        x, = ctx.saved_tensors
        sqrt_2_inv = 0.7071067811865476
        sqrt_2pi_inv = 0.3989422804014327
        cdf = 0.5 * (1.0 + torch.erf(x * sqrt_2_inv))
        pdf = sqrt_2pi_inv * torch.exp(-0.5 * x * x)
        d_gelu = cdf + x * pdf
        return grad_output * d_gelu


def demo_autograd_function():
    print("=== Section 15.2: wiring the kernel into a real graph node ===")
    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y = TritonGELU.apply(x)
    print("y.requires_grad:", y.requires_grad)
    print("y.grad_fn:", y.grad_fn)

    y.sum().backward()
    print("x.grad:", x.grad)

    x2 = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y2 = torch.nn.functional.gelu(x2, approximate="none")
    y2.sum().backward()
    print("torch GELU x2.grad:", x2.grad)
    print("gradients match?", torch.allclose(x.grad, x2.grad, atol=1e-5))


if __name__ == "__main__":
    demo_autograd_function()
