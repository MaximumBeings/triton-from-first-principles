import torch
import triton
import triton.language as tl


@triton.jit
def fma_relu_kernel(x_ptr, a, b, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = a * x + b
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_fma_relu(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    assert x.ndim == 1
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    fma_relu_kernel[(1,)](x, a, b, out, n, BLOCK=BLOCK)
    return out


def demo_fused_kernel():
    print("=== Section 12.1: a complete, assembled element-wise kernel ===")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    a, b = 2.0, -1.0
    out = fused_fma_relu(x, a, b)
    print("fused kernel result:", out.tolist())

    ref = torch.relu(a * x + b)
    print("unfused torch reference:", ref.tolist())
    print("match?", torch.equal(out, ref))


if __name__ == "__main__":
    demo_fused_kernel()
