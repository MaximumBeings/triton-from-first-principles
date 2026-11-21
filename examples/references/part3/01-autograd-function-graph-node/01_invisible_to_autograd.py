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


def plain_triton_gelu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    gelu_exact_kernel[(1,)](x, out, n, BLOCK=BLOCK)
    return out


def demo_invisible_to_autograd():
    print("=== Section 15.1: a raw kernel launch is invisible to autograd ===")
    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y = plain_triton_gelu(x)
    print("x.requires_grad:", x.requires_grad)
    print("y.requires_grad:", y.requires_grad)
    print("y.grad_fn:", y.grad_fn)
    try:
        y.sum().backward()
        print("backward succeeded (unexpected!):", x.grad)
    except Exception as e:
        print(f"y.sum().backward() -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    demo_invisible_to_autograd()
