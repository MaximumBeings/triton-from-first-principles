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


def section_18_3():
    print("=== Section 18.3: manual sweep across candidate configs ===")
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
        ok = torch.allclose(out, oracle, atol=1e-5)
        print(f"BLOCK={cfg['BLOCK']:5d} num_warps={cfg['num_warps']} num_stages={cfg['num_stages']}"
              f"  allclose to oracle? {ok}")
        outputs.append(out)

    print()
    print("BLOCK=256 output equals BLOCK=1024 output bit-for-bit?", torch.equal(outputs[0], outputs[2]))


if __name__ == "__main__":
    section_18_3()
