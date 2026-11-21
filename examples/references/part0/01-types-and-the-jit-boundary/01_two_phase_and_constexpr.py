import torch
import triton
import triton.language as tl


@triton.jit
def relu_toggle_kernel(x_ptr, out_ptr, n, USE_RELU: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    if USE_RELU:
        x = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


def demo_two_phase_model():
    print("=== Two-Phase Model: constexpr-selected kernel variants ===")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, -3.0])
    n = x.shape[0]

    out_relu = torch.empty_like(x)
    relu_toggle_kernel[(1,)](x, out_relu, n, USE_RELU=True, BLOCK=8)
    print("USE_RELU=True  ->", out_relu.tolist())

    out_norelu = torch.empty_like(x)
    relu_toggle_kernel[(1,)](x, out_norelu, n, USE_RELU=False, BLOCK=8)
    print("USE_RELU=False ->", out_norelu.tolist())


def demo_constexpr_requirement():
    print("\n=== tl.arange requires a genuine tl.constexpr bound ===")

    @triton.jit
    def bad_arange_kernel(x_ptr, out_ptr, n, BLOCK):   # no tl.constexpr annotation
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * 2.0, mask=mask)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    try:
        out = torch.empty_like(x)
        bad_arange_kernel[(1,)](x, out, 4, 4)
        print("unexpectedly succeeded:", out.tolist())
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    demo_two_phase_model()
    demo_constexpr_requirement()
