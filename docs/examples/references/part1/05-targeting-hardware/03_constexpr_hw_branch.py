import torch
import triton
import triton.language as tl


@triton.jit
def hw_aware_kernel(x_ptr, out_ptr, n, USE_FAST_PATH: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    if USE_FAST_PATH:
        result = x * 2.0
    else:
        result = x + x
    tl.store(out_ptr + offsets, result, mask=mask)


def launch_for_hardware(x, out, n, BLOCK, compute_capability):
    use_fast_path = compute_capability >= 90
    hw_aware_kernel[(1,)](x, out, n, USE_FAST_PATH=use_fast_path, BLOCK=BLOCK)


def demo_constexpr_hw_branch():
    print("=== Section 10.3: a tl.constexpr flag selecting between hardware-specific code paths ===")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    out_new = torch.empty_like(x)
    launch_for_hardware(x, out_new, 4, 4, compute_capability=90)
    print("compute_capability=90 (fast path)   ->", out_new.tolist())

    out_old = torch.empty_like(x)
    launch_for_hardware(x, out_old, 4, 4, compute_capability=80)
    print("compute_capability=80 (fallback path) ->", out_old.tolist())


if __name__ == "__main__":
    demo_constexpr_hw_branch()
