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
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    fma_relu_kernel[(1,)](x, a, b, out, n, BLOCK=BLOCK)
    return out


def demo_memory_passes():
    print("=== Section 12.2: identical results, different numbers of memory passes ===")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    a, b = 2.0, -1.0

    # naive chain: 3 separate operations, each conceptually a full read+write pass
    y1 = a * x
    y2 = y1 + b
    y3 = torch.relu(y2)
    print("naive chain result (3 separate ops):", y3.tolist())

    # fused: one kernel launch, one read of x, one write of the result
    fused = fused_fma_relu(x, a, b)
    print("fused kernel result   (1 kernel launch):", fused.tolist())
    print("identical?", torch.equal(y3, fused))


if __name__ == "__main__":
    demo_memory_passes()
