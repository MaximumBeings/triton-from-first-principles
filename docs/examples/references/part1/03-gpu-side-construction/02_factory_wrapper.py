import torch
import triton
import triton.language as tl


@triton.jit
def fill_kernel(out_ptr, value, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, value, mask=mask)


def triton_full(shape, value, dtype=torch.float32, device="cpu"):
    out = torch.empty(shape, dtype=dtype, device=device)
    n = out.numel()
    BLOCK = triton.next_power_of_2(n)
    fill_kernel[(1,)](out, value, n, BLOCK=BLOCK)
    return out


def demo_factory_wrapper():
    print("=== Section 8.2: a hand-written 'factory function' wrapper ===")
    t = triton_full((6,), 7.0)
    print("triton_full((6,), 7.0) ->", t.tolist())

    print("\ncompare against just using torch's own factory function directly:")
    t2 = torch.full((6,), 7.0)
    print("torch.full((6,), 7.0)  ->", t2.tolist())


if __name__ == "__main__":
    demo_factory_wrapper()
