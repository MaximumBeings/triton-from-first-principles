"""
Chapter 16 standalone: writing, generalizing, wiring, and measuring the
precision of a genuine Triton backward kernel for GELU.

Deliberately excludes the mask-omission crash from Worked Example 16.2.2 --
that demonstration kills the process outright (a real, reproducible
segmentation fault under TRITON_INTERPRET=1), so it cannot be safely
combined with the rest of this file's output. See the chapter text for
that demonstration's genuinely captured result, run in isolation.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Section 16.1: the backward kernel, and the chain-rule trap
# ---------------------------------------------------------------------------

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


def section_16_1():
    print("=== Section 16.1: invisible at grad_output=1, exposed at grad_output=2 ===")
    x = torch.tensor([-1.0, 0.5, 2.0])
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)

    for label, grad_output in [("grad_output=1.0", torch.ones_like(x)),
                                ("grad_output=2.0", torch.full_like(x, 2.0))]:
        x_oracle = x.clone().requires_grad_(True)
        y_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
        y_oracle.backward(grad_output)

        correct_grad = torch.empty_like(x)
        gelu_backward_kernel[(1,)](grad_output, x, correct_grad, n, BLOCK=BLOCK)
        buggy_grad = torch.empty_like(x)
        gelu_backward_kernel_buggy_no_chain_rule[(1,)](grad_output, x, buggy_grad, n, BLOCK=BLOCK)

        print(f"{label}: correct matches oracle? {torch.allclose(correct_grad, x_oracle.grad)}"
              f"  buggy matches oracle? {torch.allclose(buggy_grad, x_oracle.grad)}")


# ---------------------------------------------------------------------------
# Section 16.2: masked, multi-program generalization
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


def triton_gelu_forward(x, BLOCK=1024):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    gelu_forward_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out


def triton_gelu_backward_plain(grad_output, x, BLOCK=1024):
    grad_input = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    gelu_backward_kernel[grid](grad_output, x, grad_input, n, BLOCK=BLOCK)
    return grad_input


def section_16_2():
    print("=== Section 16.2: n=10000, BLOCK=1024, 10 programs (last one masked) ===")
    torch.manual_seed(0)
    n = 10000
    BLOCK = 1024
    x = torch.randn(n)
    out = triton_gelu_forward(x, BLOCK=BLOCK)
    oracle_out = torch.nn.functional.gelu(x, approximate="none")
    print("forward allclose to oracle?", torch.allclose(out, oracle_out, atol=1e-5))

    grad_output = torch.randn(n)
    grad_input = triton_gelu_backward_plain(grad_output, x, BLOCK=BLOCK)
    x_oracle = x.clone().requires_grad_(True)
    y_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    y_oracle.backward(grad_output)
    print("backward allclose to oracle?", torch.allclose(grad_input, x_oracle.grad, atol=1e-5))


# ---------------------------------------------------------------------------
# Section 16.3: wired into TritonGELU, plus fp16 vs. fp32 storage precision
# ---------------------------------------------------------------------------

@triton.jit
def gelu_backward_kernel_upcast(grad_out_ptr, x_ptr, grad_in_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    SQRT_2PI_INV: tl.constexpr = 0.3989422804014327
    cdf = 0.5 * (1.0 + tl.math.erf(x * SQRT_2_INV))
    pdf = SQRT_2PI_INV * tl.exp(-0.5 * x * x)
    d_gelu = cdf + x * pdf
    grad_in = grad_out * d_gelu
    tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


def triton_gelu_backward(grad_output, x, BLOCK=1024):
    x = x.contiguous()
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    gelu_backward_kernel_upcast[grid](grad_output, x, grad_input, n, BLOCK=BLOCK)
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
        return triton_gelu_backward(grad_output.contiguous(), x)


def section_16_3():
    print("=== Section 16.3: TritonGELU wired end-to-end, n=5000 ===")
    torch.manual_seed(1)
    n = 5000
    x = torch.randn(n, requires_grad=True)
    y = TritonGELU.apply(x)
    y.sum().backward()

    x_oracle = x.detach().clone().requires_grad_(True)
    y_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    y_oracle.sum().backward()

    print("forward allclose to oracle?", torch.allclose(y.detach(), y_oracle.detach(), atol=1e-5))
    print("x.grad allclose to oracle? ", torch.allclose(x.grad, x_oracle.grad, atol=1e-5))

    print()
    print("=== Section 16.3: fp16 vs. fp32 storage precision (fp32 math throughout) ===")
    torch.manual_seed(2)
    n2, BLOCK2 = 2048, 512
    x_fp32 = torch.randn(n2) * 3.0
    grad_out_fp32 = torch.randn(n2)

    x64 = x_fp32.double().requires_grad_(True)
    y64 = torch.nn.functional.gelu(x64, approximate="none")
    y64.backward(grad_out_fp32.double())
    ground_truth = x64.grad

    grad_in_fp32 = triton_gelu_backward(grad_out_fp32, x_fp32, BLOCK=BLOCK2)
    diff_fp32 = (grad_in_fp32.double() - ground_truth).abs().max().item()
    print("fp32-in/fp32-out max abs diff from fp64 ground truth:", diff_fp32)

    grad_in_fp16 = triton_gelu_backward(grad_out_fp32.half(), x_fp32.half(), BLOCK=BLOCK2)
    diff_fp16 = (grad_in_fp16.double() - ground_truth).abs().max().item()
    print("fp16-in/fp16-out (fp32 math internally) max abs diff:", diff_fp16)
    print("ratio (fp16 path error / fp32 path error):", diff_fp16 / diff_fp32)


if __name__ == "__main__":
    section_16_1()
    print()
    section_16_2()
    print()
    section_16_3()
