import torch
import triton
import triton.language as tl


@triton.jit
def introspect_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.static_print("x_ptr type:", x_ptr)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def shape_probe_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    tl.static_print("x_ptr.shape as seen inside the kernel:", x_ptr.shape)
    tl.store(out_ptr, 0.0)


def demo_pointer_boundary():
    print("=== Section 6.1: what actually crosses into the kernel ===")
    x = torch.randn(4)
    out = torch.empty(4)
    print("host-side: x.shape =", x.shape, " x.stride() =", x.stride(), " x.dtype =", x.dtype)
    introspect_kernel[(1,)](x, out, 4, BLOCK=4)

    shape_probe_kernel[(1,)](x, out, BLOCK=4)
    print("(the real tensor's shape is [4] -- compare against what the kernel saw above)")


if __name__ == "__main__":
    demo_pointer_boundary()
