import torch
import triton
import triton.language as tl


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
    print("=== Section 16.1: the backward kernel vs. Chapter 15's plain-PyTorch formula ===")
    x = torch.tensor([-1.0, 0.5, 2.0])
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)

    grad_output_ones = torch.ones_like(x)
    x_oracle1 = x.clone().requires_grad_(True)
    y_oracle1 = torch.nn.functional.gelu(x_oracle1, approximate="none")
    y_oracle1.backward(grad_output_ones)

    correct_grad1 = torch.empty_like(x)
    gelu_backward_kernel[(1,)](grad_output_ones, x, correct_grad1, n, BLOCK=BLOCK)
    buggy_grad1 = torch.empty_like(x)
    gelu_backward_kernel_buggy_no_chain_rule[(1,)](grad_output_ones, x, buggy_grad1, n, BLOCK=BLOCK)

    print("with grad_output=1.0 (both look correct):")
    print("  correct kernel matches oracle?", torch.allclose(correct_grad1, x_oracle1.grad))
    print("  buggy kernel matches oracle?  ", torch.allclose(buggy_grad1, x_oracle1.grad))

    grad_output_twos = torch.full_like(x, 2.0)
    x_oracle2 = x.clone().requires_grad_(True)
    y_oracle2 = torch.nn.functional.gelu(x_oracle2, approximate="none")
    y_oracle2.backward(grad_output_twos)

    correct_grad2 = torch.empty_like(x)
    gelu_backward_kernel[(1,)](grad_output_twos, x, correct_grad2, n, BLOCK=BLOCK)
    buggy_grad2 = torch.empty_like(x)
    gelu_backward_kernel_buggy_no_chain_rule[(1,)](grad_output_twos, x, buggy_grad2, n, BLOCK=BLOCK)

    print("with grad_output=2.0 (the trap is exposed):")
    print("  correct kernel matches oracle?", torch.allclose(correct_grad2, x_oracle2.grad))
    print("  buggy kernel matches oracle?  ", torch.allclose(buggy_grad2, x_oracle2.grad))
    print("  buggy kernel is exactly half? ", torch.allclose(buggy_grad2 * 2.0, x_oracle2.grad))


if __name__ == "__main__":
    section_16_1()
