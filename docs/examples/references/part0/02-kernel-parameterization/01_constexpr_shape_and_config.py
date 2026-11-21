import torch
import triton
import triton.language as tl


@triton.jit
def shape_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    tl.static_print("COMPILING a version with BLOCK =", BLOCK)
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)


def demo_constexpr_shape():
    print("=== Section 2.1: two constexpr signatures = two compiled artifacts ===")
    x = torch.arange(8, dtype=torch.float32)
    out8 = torch.empty_like(x)
    shape_kernel[(1,)](x, out8, 8, BLOCK=8)
    print("BLOCK=8 result:", out8.tolist())
    out16 = torch.empty_like(x)
    shape_kernel[(1,)](x, out16, 8, BLOCK=16)
    print("BLOCK=16 result:", out16.tolist())


def demo_config_objects():
    print("\n=== Section 2.2: triton.Config objects, genuinely constructed ===")
    configs = [
        triton.Config(kwargs={"BLOCK": 64}, num_warps=2),
        triton.Config(kwargs={"BLOCK": 128}, num_warps=4),
        triton.Config(kwargs={"BLOCK": 256}, num_warps=8),
    ]
    for c in configs:
        print("kwargs:", c.kwargs, "num_warps:", c.num_warps, "num_stages:", c.num_stages)


if __name__ == "__main__":
    demo_constexpr_shape()
    demo_config_objects()
