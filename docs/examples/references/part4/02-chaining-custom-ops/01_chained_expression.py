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
def gelu_backward_kernel_buggy_no_chain_rule(grad_out_ptr, x_ptr, grad_in_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    SQRT_2PI_INV: tl.constexpr = 0.3989422804014327
    cdf = 0.5 * (1.0 + tl.math.erf(x * SQRT_2_INV))
    pdf = SQRT_2PI_INV * tl.exp(-0.5 * x * x)
    d_gelu = cdf + x * pdf
    tl.store(grad_in_ptr + offsets, d_gelu, mask=mask)  # BUG: never multiplied by grad_out


def triton_gelu_forward(x, BLOCK=1024):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    gelu_forward_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out


def _launch_bwd(kernel, grad_output, x, BLOCK=1024):
    grad_input = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](grad_output, x, grad_input, n, BLOCK=BLOCK)
    return grad_input


class TritonGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = triton_gelu_forward(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return _launch_bwd(gelu_backward_kernel, grad_output.contiguous(), x)


class TritonGELUBuggy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = triton_gelu_forward(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return _launch_bwd(gelu_backward_kernel_buggy_no_chain_rule, grad_output.contiguous(), x)


def section_17_1():
    print("=== Section 17.1: TritonGELU chained inside w = 2*z + 1, loss = w.sum() ===")
    torch.manual_seed(3)
    x = torch.randn(6, requires_grad=True)

    captured_grad_output = {}
    z = TritonGELU.apply(x)
    z.register_hook(lambda grad: captured_grad_output.setdefault("z", grad.clone()))
    w = 2 * z + 1
    loss = w.sum()
    loss.backward()

    print("grad_output that arrived at TritonGELU's node (dL/dz):", captured_grad_output["z"])

    x_oracle = x.detach().clone().requires_grad_(True)
    z_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    (2 * z_oracle + 1).sum().backward()
    print("x.grad allclose to oracle?", torch.allclose(x.grad, x_oracle.grad, atol=1e-5))

    print()
    print("=== Section 17.1: the buggy kernel inside the SAME chain, no hand-injected grad_output ===")
    x2 = x.detach().clone().requires_grad_(True)
    z2 = TritonGELUBuggy.apply(x2)
    loss2 = (2 * z2 + 1).sum()
    loss2.backward()
    print("buggy chain matches oracle?", torch.allclose(x2.grad, x_oracle.grad, atol=1e-5))


if __name__ == "__main__":
    section_17_1()
