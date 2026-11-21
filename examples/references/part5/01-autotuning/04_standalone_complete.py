"""
Chapter 18 standalone: the shape of an autotuning search space, the genuine
failure of @triton.autotune itself in a driver-less environment, and what a
manual config sweep can still verify without any hardware timing at all.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gelu_forward_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    cdf = 0.5 * (1.0 + tl.math.erf(x * SQRT_2_INV))
    out = x * cdf
    tl.store(out_ptr + offsets, out, mask=mask)


configs = [
    triton.Config({"BLOCK": 256}, num_warps=2),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=8),
]


@triton.autotune(configs=configs, key=["n"])
@triton.jit
def gelu_forward_kernel_autotuned(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    cdf = 0.5 * (1.0 + tl.math.erf(x * SQRT_2_INV))
    out = x * cdf
    tl.store(out_ptr + offsets, out, mask=mask)


def section_18_1():
    print("=== Section 18.1: triton.Config objects and an illegal candidate ===")
    cfg = triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3)
    print("kwargs:", cfg.kwargs, " num_warps:", cfg.num_warps, " num_stages:", cfg.num_stages)

    torch.manual_seed(0)
    n = 4096
    x = torch.randn(n)
    try:
        out_bad = torch.empty_like(x)
        gelu_forward_kernel[(triton.cdiv(n, 300),)](x, out_bad, n, BLOCK=300, num_warps=4, num_stages=2)
        print("BLOCK=300 ran without error (unexpected)")
    except Exception as e:
        print(f"BLOCK=300 caught, as expected: {type(e).__name__}")


def section_18_2():
    print()
    print("=== Section 18.2: attempting @triton.autotune itself ===")
    torch.manual_seed(0)
    n = 4096
    x = torch.randn(n)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    try:
        gelu_forward_kernel_autotuned[grid](x, out, n)
        print("ran without error (unexpected in this environment)")
    except RuntimeError as e:
        print(f"caught, as expected: {type(e).__name__}: {e}")


def section_18_3():
    print()
    print("=== Section 18.3: manual sweep -- correctness invariance across configs ===")
    torch.manual_seed(0)
    n = 4096
    x = torch.randn(n)
    oracle = torch.nn.functional.gelu(x, approximate="none")

    candidate_configs = [
        {"BLOCK": 256, "num_warps": 2, "num_stages": 1},
        {"BLOCK": 512, "num_warps": 4, "num_stages": 2},
        {"BLOCK": 1024, "num_warps": 8, "num_stages": 3},
        {"BLOCK": 2048, "num_warps": 8, "num_stages": 4},
    ]

    outputs = []
    for cfg in candidate_configs:
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, cfg["BLOCK"]),)
        gelu_forward_kernel[grid](x, out, n, BLOCK=cfg["BLOCK"],
                                  num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
        outputs.append(out)
        print(f"BLOCK={cfg['BLOCK']:5d} allclose to oracle? {torch.allclose(out, oracle, atol=1e-5)}")

    print("all four candidates bit-for-bit identical to each other?",
          all(torch.equal(outputs[0], o) for o in outputs[1:]))


if __name__ == "__main__":
    section_18_1()
    section_18_2()
    section_18_3()
