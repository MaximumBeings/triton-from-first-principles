import torch
import triton
import triton.language as tl


@triton.jit
def hinted_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    offsets = tl.multiple_of(offsets, BLOCK)
    offsets = tl.max_contiguous(offsets, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


@triton.jit
def lying_multiple_of_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK) + 3
    offsets = tl.multiple_of(offsets, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


def demo_alignment_hints():
    print("=== Section 7.2: tl.multiple_of / tl.max_contiguous, told the truth ===")
    n = 16
    x = torch.arange(n, dtype=torch.float32)
    out = torch.empty(n)
    BLOCK = 8
    grid = (triton.cdiv(n, BLOCK),)
    hinted_kernel[grid](x, out, n, BLOCK=BLOCK)
    print("hinted kernel result:", out.tolist())
    print("matches x*2?", torch.equal(out, x * 2))

    print("\n=== a false tl.multiple_of hint -- costs nothing under the interpreter ===")
    out2 = torch.zeros(n)
    lying_multiple_of_kernel[(1,)](x, out2, n, BLOCK=8)
    print("lied multiple_of result:", out2.tolist())
    print("(no error raised -- multiple_of/max_contiguous are unchecked compiler hints, not assertions)")


if __name__ == "__main__":
    demo_alignment_hints()
