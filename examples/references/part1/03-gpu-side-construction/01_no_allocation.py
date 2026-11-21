import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def local_zeros_kernel(out_ptr, BLOCK: tl.constexpr):
    local = tl.zeros((BLOCK,), dtype=tl.float32)
    local = local + 1.0
    tl.store(out_ptr + tl.arange(0, BLOCK), local)


def demo_preallocated_output():
    print("=== Section 8.1: the kernel needs somewhere to write BEFORE it launches ===")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0])
    out = torch.empty_like(x)
    add_kernel[(1,)](x, y, out, 4, BLOCK=4)
    print("out (pre-allocated by torch.empty_like, then filled by the kernel):", out.tolist())

    try:
        bad_out = None
        add_kernel[(1,)](x, y, bad_out, 4, BLOCK=4)
        print("unexpectedly succeeded")
    except Exception as e:
        print(f"passing None as the output buffer -> {type(e).__name__}: {e}")


def demo_tl_zeros_is_not_allocation():
    print("\n=== Section 8.1: tl.zeros() is a register-resident value, not a new allocation ===")
    out2 = torch.empty(4)
    local_zeros_kernel[(1,)](out2, BLOCK=4)
    print("value only survives because it was explicitly tl.store'd:", out2.tolist())


if __name__ == "__main__":
    demo_preallocated_output()
    demo_tl_zeros_is_not_allocation()
