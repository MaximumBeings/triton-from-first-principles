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
    print("y.grad_fn:", y.grad_fn)
    y.sum().backward()

    x_oracle = x.detach().clone().requires_grad_(True)
    y_oracle = torch.nn.functional.gelu(x_oracle, approximate="none")
    y_oracle.sum().backward()

    print("forward allclose to oracle?", torch.allclose(y.detach(), y_oracle.detach(), atol=1e-5))
    print("x.grad allclose to oracle? ", torch.allclose(x.grad, x_oracle.grad, atol=1e-5))

    print()
    print("=== Section 16.3: fp16 storage precision vs. fp32, with fp32 math throughout ===")
    torch.manual_seed(2)
    n2 = 2048
    BLOCK2 = 512
    x_fp32 = torch.randn(n2) * 3.0
    grad_out_fp32 = torch.randn(n2)

    x64 = x_fp32.double().requires_grad_(True)
    y64 = torch.nn.functional.gelu(x64, approximate="none")
    y64.backward(grad_out_fp32.double())
    ground_truth = x64.grad

    grad_in_fp32 = triton_gelu_backward(grad_out_fp32, x_fp32, BLOCK=BLOCK2)
    diff_fp32 = (grad_in_fp32.double() - ground_truth).abs().max().item()
    print("fp32-in/fp32-out max abs diff from fp64 ground truth:", diff_fp32)

    x_fp16 = x_fp32.half()
    grad_out_fp16 = grad_out_fp32.half()
    grad_in_fp16 = triton_gelu_backward(grad_out_fp16, x_fp16, BLOCK=BLOCK2)
    diff_fp16 = (grad_in_fp16.double() - ground_truth).abs().max().item()
    print("fp16-in/fp16-out (fp32 math internally) max abs diff:", diff_fp16)
    print("ratio (fp16 path error / fp32 path error):", diff_fp16 / diff_fp32)

    print()
    print("=== Section 16.3: tl.math.erf refuses fp16 operands without the upcast ===")
    try:
        grad_in_bad = torch.empty(n2, dtype=torch.float16)
        grid = (triton.cdiv(n2, BLOCK2),)

        @triton.jit
        def _gelu_backward_no_upcast(grad_out_ptr, x_ptr, grad_in_ptr, n, BLOCK: tl.constexpr):
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

        _gelu_backward_no_upcast[grid](grad_out_fp16, x_fp16, grad_in_bad, n2, BLOCK=BLOCK2)
        print("no exception raised (unexpected)")
    except Exception as e:
        print(f"caught exception, as expected: {type(e).__name__}: {e}")


if __name__ == "__main__":
    section_16_3()
