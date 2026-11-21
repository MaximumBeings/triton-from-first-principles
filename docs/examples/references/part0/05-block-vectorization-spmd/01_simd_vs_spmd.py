import torch
import triton
import triton.language as tl


@triton.jit
def double_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


@triton.jit
def double_kernel_grid(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


def demo_power_of_two():
    print("=== Section 5.1: BLOCK must be a power of 2 ===")
    x7 = torch.arange(7, dtype=torch.float32)
    out7 = torch.empty(7)
    try:
        double_kernel[(1,)](x7, out7, 7, BLOCK=7)
        print("BLOCK=7 succeeded:", out7.tolist())
    except Exception as e:
        print(f"BLOCK=7 -> {type(e).__name__}: {e}")

    out7b = torch.empty(7)
    double_kernel[(1,)](x7, out7b, 7, BLOCK=8)
    print("BLOCK=8 (next power of 2, masked down to n=7):", out7b.tolist())


def demo_no_remainder_loop():
    print("\n=== Section 5.2: one masked kernel handles the 'remainder' automatically ===")
    n, BLOCK = 10, 8
    grid = (triton.cdiv(n, BLOCK),)
    x10 = torch.arange(10, dtype=torch.float32)
    out10 = torch.empty(10)
    double_kernel_grid[grid](x10, out10, n, BLOCK=BLOCK)
    print(f"n={n}, BLOCK={BLOCK}, grid={grid} -> one kernel, no separate remainder kernel")
    print("result:", out10.tolist(), "correct?", torch.equal(out10, x10 * 2))


if __name__ == "__main__":
    demo_power_of_two()
    demo_no_remainder_loop()
