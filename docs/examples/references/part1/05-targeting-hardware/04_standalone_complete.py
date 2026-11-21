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


def main():
    print("=== Section 10.1: hardware queries without a GPU ===")
    try:
        triton.runtime.driver.active.get_current_target()
    except Exception as e:
        print(f"hardware query -> {type(e).__name__}: {e}")

    print("\n=== Section 10.2: num_warps / num_stages ===")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0])
    for nw in [1, 4, 8]:
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, 4, BLOCK=4, num_warps=nw)
        print(f"num_warps={nw} -> {out.tolist()}")

    print("\n=== Section 10.3: constexpr hardware branching ===")
    out_new = torch.empty_like(x)
    launch_for_hardware(x, out_new, 4, 4, compute_capability=90)
    print("compute_capability=90 ->", out_new.tolist())
    out_old = torch.empty_like(x)
    launch_for_hardware(x, out_old, 4, 4, compute_capability=80)
    print("compute_capability=80 ->", out_old.tolist())


if __name__ == "__main__":
    main()
