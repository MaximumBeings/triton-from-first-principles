import torch


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


def demo_safe_save_for_backward():
    print("=== Section 15.3: save_for_backward detects the in-place mutation ===")
    x = torch.tensor([3.0], requires_grad=True)
    y = SafeSquare.apply(x)
    with torch.no_grad():
        x.mul_(10.0)
    try:
        y.sum().backward()
        print("backward succeeded (unexpected!):", x.grad)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


def demo_unsafe_ctx_attribute():
    print("\n=== Section 15.3: ctx.x bypasses the safety check entirely ===")
    x2 = torch.tensor([3.0], requires_grad=True)
    y2 = UnsafeSquare.apply(x2)
    with torch.no_grad():
        x2.mul_(10.0)
    y2.sum().backward()
    print("backward succeeded silently, x2.grad:", x2.grad)
    print("(correct gradient at x=3 would be 6.0 -- this used the mutated x=30 instead)")


if __name__ == "__main__":
    demo_safe_save_for_backward()
    demo_unsafe_ctx_attribute()
