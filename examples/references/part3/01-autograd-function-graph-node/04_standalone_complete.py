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


def plain_triton_gelu(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    gelu_exact_kernel[(1,)](x, out, n, BLOCK=BLOCK)
    return out


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
        x, = ctx.saved_tensors
        sqrt_2_inv = 0.7071067811865476
        sqrt_2pi_inv = 0.3989422804014327
        cdf = 0.5 * (1.0 + torch.erf(x * sqrt_2_inv))
        pdf = sqrt_2pi_inv * torch.exp(-0.5 * x * x)
        d_gelu = cdf + x * pdf
        return grad_output * d_gelu


class SafeSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x


class UnsafeSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.x
        return grad_output * 2 * x


def main():
    print("=== Section 15.1: invisible to autograd ===")
    x1 = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y1 = plain_triton_gelu(x1)
    print("y.grad_fn:", y1.grad_fn)
    try:
        y1.sum().backward()
    except Exception as e:
        print(f"backward -> {type(e).__name__}: {e}")

    print("\n=== Section 15.2: wired into a real graph node ===")
    x2 = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    y2 = TritonGELU.apply(x2)
    print("y.grad_fn:", y2.grad_fn)
    y2.sum().backward()
    x2_ref = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    torch.nn.functional.gelu(x2_ref, approximate="none").sum().backward()
    print("gradients match torch's own GELU?", torch.allclose(x2.grad, x2_ref.grad, atol=1e-5))

    print("\n=== Section 15.3: save_for_backward vs. ctx.x ===")
    xs = torch.tensor([3.0], requires_grad=True)
    ys = SafeSquare.apply(xs)
    with torch.no_grad():
        xs.mul_(10.0)
    try:
        ys.sum().backward()
    except Exception as e:
        print(f"safe version -> {type(e).__name__} (mutation caught)")

    xu = torch.tensor([3.0], requires_grad=True)
    yu = UnsafeSquare.apply(xu)
    with torch.no_grad():
        xu.mul_(10.0)
    yu.sum().backward()
    print("unsafe version -> silently wrong grad:", xu.grad.item(), "(correct would be 6.0)")


if __name__ == "__main__":
    main()
