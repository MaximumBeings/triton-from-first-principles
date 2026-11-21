import torch
import triton
import triton.language as tl


@triton.jit
def pid_branch_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    if pid == 0:
        x = x * 100.0
    else:
        x = x + 1.0
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def bad_block_branch_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    if mask:                      # BLOCK-shaped, not one bool -- see Worked Example 1.4.2
        x = x * 2.0
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def correct_block_conditional_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.where(mask, x * 2.0, x)   # the fix: a per-lane answer, not one collapsed answer
    tl.store(out_ptr + offsets, x, mask=mask)


def demo_scalar_branch():
    print("=== Legal: branching on a genuine scalar (tl.program_id) ===")
    x = torch.arange(6, dtype=torch.float32)
    out = torch.empty_like(x)
    pid_branch_kernel[(2,)](x, out, 6, BLOCK=4)
    print("result:", out.tolist())


def demo_block_branch_trap():
    print("\n=== The trap: branching on a whole block of booleans ===")
    x = torch.arange(6, dtype=torch.float32)
    out_bad = torch.empty_like(x)
    try:
        bad_block_branch_kernel[(2,)](x, out_bad, 6, BLOCK=4)
        print("ran without raising, but silently wrong:", out_bad.tolist())
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

    out_correct = torch.empty_like(x)
    correct_block_conditional_kernel[(2,)](x, out_correct, 6, BLOCK=4)
    print("tl.where fix, correct per-lane result:", out_correct.tolist())


if __name__ == "__main__":
    demo_scalar_branch()
    demo_block_branch_trap()
