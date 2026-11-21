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


@triton.jit
def fused_activation_kernel(x_ptr, out_ptr, n, ACTIVATION: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    if ACTIVATION == "relu":
        x = tl.maximum(x, 0.0)
    elif ACTIVATION == "gelu":
        x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offsets, x, mask=mask)


def run_with_scratch(x: torch.Tensor) -> torch.Tensor:
    """Section 2.4: PyTorch owns `out`'s memory; the kernel never allocates anything."""
    out = torch.empty_like(x)
    shape_kernel[(1,)](x, out, x.shape[0], BLOCK=x.shape[0])
    return out


def main():
    print("=== Section 2.1: two constexpr signatures = two compiled artifacts ===")
    x = torch.arange(8, dtype=torch.float32)
    out8 = torch.empty_like(x)
    shape_kernel[(1,)](x, out8, 8, BLOCK=8)
    print("BLOCK=8 result:", out8.tolist())
    out16 = torch.empty_like(x)
    shape_kernel[(1,)](x, out16, 8, BLOCK=16)
    print("BLOCK=16 result:", out16.tolist())

    print("\n=== Section 2.2: triton.Config objects, genuinely constructed ===")
    configs = [
        triton.Config(kwargs={"BLOCK": 64}, num_warps=2),
        triton.Config(kwargs={"BLOCK": 128}, num_warps=4),
        triton.Config(kwargs={"BLOCK": 256}, num_warps=8),
    ]
    for c in configs:
        print("kwargs:", c.kwargs, "num_warps:", c.num_warps, "num_stages:", c.num_stages)

    print("\n=== Section 2.4: scratch memory belongs to PyTorch, not the kernel ===")
    scratch_out = run_with_scratch(torch.arange(4, dtype=torch.float32))
    print("run_with_scratch result:", scratch_out.tolist())

    print("\n=== Section 2.5: one kernel body, three constexpr-selected specializations ===")
    xv = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    n = xv.shape[0]
    for act in ["none", "relu", "gelu"]:
        outv = torch.empty_like(xv)
        fused_activation_kernel[(1,)](xv, outv, n, ACTIVATION=act, BLOCK=8)
        print(f"ACTIVATION={act!r:8s} ->", [round(v, 6) for v in outv.tolist()])
    print("gelu reference (torch.nn.functional.gelu):",
          [round(v, 6) for v in torch.nn.functional.gelu(xv).tolist()])


if __name__ == "__main__":
    main()
