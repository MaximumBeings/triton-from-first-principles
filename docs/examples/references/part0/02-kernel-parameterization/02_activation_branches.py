import torch
import triton
import triton.language as tl


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


def demo_activation_branches():
    print("=== Section 2.5: one kernel body, three constexpr-selected specializations ===")
    xv = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    n = xv.shape[0]
    for act in ["none", "relu", "gelu"]:
        outv = torch.empty_like(xv)
        fused_activation_kernel[(1,)](xv, outv, n, ACTIVATION=act, BLOCK=8)
        print(f"ACTIVATION={act!r:8s} ->", [round(v, 6) for v in outv.tolist()])
    print("gelu reference (torch.nn.functional.gelu):",
          [round(v, 6) for v in torch.nn.functional.gelu(xv).tolist()])


if __name__ == "__main__":
    demo_activation_branches()
