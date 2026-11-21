import torch
import triton
import triton.language as tl


@triton.jit
def global_id_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, offsets)


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def demo_program_hierarchy():
    print("=== Section 4.1: grid of 4 programs, BLOCK=8 ===")
    out = torch.empty(32, dtype=torch.int32)
    global_id_kernel[(4,)](out, BLOCK=8)
    print("all global ids:", out.tolist())
    print("program 2's slice (expect 16..23):", out[16:24].tolist())


def demo_ceiling_division_grid():
    print("\n=== Section 4.2: ceiling-division grid, 1000 elements, BLOCK=256 ===")
    n, BLOCK = 1000, 256
    grid = (triton.cdiv(n, BLOCK),)
    print("triton.cdiv(1000, 256) =", triton.cdiv(n, BLOCK), "-> grid =", grid)
    a = torch.arange(n, dtype=torch.float32)
    b = torch.arange(n, dtype=torch.float32) * 2
    out = torch.empty(n, dtype=torch.float32)
    add_kernel[grid](a, b, out, n, BLOCK=BLOCK)
    print("correct?", torch.equal(out, a + b))
    total = grid[0] * BLOCK
    print(f"total thread-slots launched: {grid[0]} programs x {BLOCK} = {total}, "
          f"needed: {n}, idle (masked off): {total - n}")


if __name__ == "__main__":
    demo_program_hierarchy()
    demo_ceiling_division_grid()
