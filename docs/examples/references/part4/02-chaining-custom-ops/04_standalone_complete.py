"""
Chapter 17 standalone: chaining a custom autograd.Function inside ordinary
PyTorch expressions, reading the resulting graph structure directly, and
confirming how grad_output is assembled when a value has more than one
consumer.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Shared kernels: GELU forward/backward (correct and buggy), and square
# ---------------------------------------------------------------------------

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


call_order = []
captured_grads = []


class TritonGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = _launch(gelu_forward_kernel, x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        call_order.append("TritonGELU.backward")
        captured_grads.append(grad_output.clone())
        x, = ctx.saved_tensors
        return _launch_bwd(gelu_backward_kernel, grad_output.contiguous(), x)


class TritonGELUBuggy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = _launch(gelu_forward_kernel, x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return _launch_bwd(gelu_backward_kernel_buggy_no_chain_rule, grad_output.contiguous(), x)


class TritonSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = _launch(square_forward_kernel, x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        call_order.append("TritonSquare.backward")
        x, = ctx.saved_tensors
        return _launch_bwd(square_backward_kernel, grad_output.contiguous(), x)


def section_17_1():
    print("=== Section 17.1: chained expression, and the chain-rule bug surfacing on its own ===")
    torch.manual_seed(3)
    x = torch.randn(6, requires_grad=True)

    hook_captured = {}
    z = TritonGELU.apply(x)
    z.register_hook(lambda grad: hook_captured.setdefault("z", grad.clone()))
    loss = (2 * z + 1).sum()
    loss.backward()
    print("grad_output at TritonGELU's node:", hook_captured["z"])

    x_oracle = x.detach().clone().requires_grad_(True)
    z_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    (2 * z_oracle + 1).sum().backward()
    print("correct chain x.grad allclose to oracle?", torch.allclose(x.grad, x_oracle.grad, atol=1e-5))

    x2 = x.detach().clone().requires_grad_(True)
    z2 = TritonGELUBuggy.apply(x2)
    (2 * z2 + 1).sum().backward()
    print("buggy chain x.grad allclose to oracle?  ", torch.allclose(x2.grad, x_oracle.grad, atol=1e-5))


def section_17_2():
    print()
    print("=== Section 17.2: graph structure and backward call order ===")
    torch.manual_seed(4)
    x = torch.randn(4, requires_grad=True)
    z = TritonGELU.apply(x)
    y = TritonSquare.apply(z)

    print("y.grad_fn is TritonSquareBackward?", "TritonSquareBackward" in str(y.grad_fn))
    print("z.grad_fn is TritonGELUBackward?  ", "TritonGELUBackward" in str(z.grad_fn))
    print("y.grad_fn.next_functions[0][0] is z.grad_fn?",
          y.grad_fn.next_functions[0][0] is z.grad_fn)

    call_order.clear()
    y.sum().backward()
    print("backward call order:", call_order)

    x_oracle = x.detach().clone().requires_grad_(True)
    z_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    (z_oracle * z_oracle).sum().backward()
    print("x.grad allclose to oracle?", torch.allclose(x.grad, x_oracle.grad, atol=1e-5))


def section_17_3():
    print()
    print("=== Section 17.3: fan-in accumulation ===")
    torch.manual_seed(5)
    x1 = torch.randn(4, requires_grad=True)

    captured_grads.clear()
    z1 = TritonGELU.apply(x1)
    z1.sum().backward()
    print("used once  -> grad_output:", captured_grads[-1])

    captured_grads.clear()
    x2 = x1.detach().clone().requires_grad_(True)
    z2 = TritonGELU.apply(x2)
    (z2.sum() + z2.sum()).backward()
    print("used twice -> grad_output:", captured_grads[-1], " (backward called", len(captured_grads), "time(s))")

    captured_grads.clear()
    x3 = x1.detach().clone().requires_grad_(True)
    z3 = TritonGELU.apply(x3)
    ((2 * z3).sum() + (3 * z3).sum()).backward()
    print("two different consumers (2x + 3x) -> grad_output:", captured_grads[-1])

    x_oracle = x1.detach().clone().requires_grad_(True)
    z_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    ((2 * z_oracle).sum() + (3 * z_oracle).sum()).backward()
    print("x3.grad allclose to oracle?", torch.allclose(x3.grad, x_oracle.grad, atol=1e-5))


if __name__ == "__main__":
    section_17_1()
    section_17_2()
    section_17_3()
