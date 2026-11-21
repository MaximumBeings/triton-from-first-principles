import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


def demo_register_pressure():
    print("=== Section 11.2: does a huge BLOCK size raise any complaint under the interpreter? ===")
    for BLOCK in [4, 4096, 65536]:
        n = BLOCK
        x = torch.arange(n, dtype=torch.float32)
        out = torch.empty(n)
        copy_kernel[(1,)](x, out, n, BLOCK=BLOCK)
        print(f"BLOCK={BLOCK:>6} -> correct? {torch.equal(out, x)}  (no resource error of any kind)")


if __name__ == "__main__":
    demo_register_pressure()
