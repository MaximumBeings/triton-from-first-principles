import torch
import triton
import triton.language as tl


@triton.jit
def dtype_report_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.static_print("compiled specialization for dtype:", x.dtype)
    tl.store(out_ptr + offsets, x, mask=mask)


def demo_dtype_specialization():
    print("=== Section 6.3: dtype triggers its own specialization, automatically ===")
    xf32 = torch.randn(4, dtype=torch.float32)
    outf32 = torch.empty(4, dtype=torch.float32)
    dtype_report_kernel[(1,)](xf32, outf32, 4, BLOCK=4)

    xf16 = torch.randn(4, dtype=torch.float16)
    outf16 = torch.empty(4, dtype=torch.float16)
    dtype_report_kernel[(1,)](xf16, outf16, 4, BLOCK=4)

    xi32 = torch.arange(4, dtype=torch.int32)
    outi32 = torch.empty(4, dtype=torch.int32)
    dtype_report_kernel[(1,)](xi32, outi32, 4, BLOCK=4)


if __name__ == "__main__":
    demo_dtype_specialization()
