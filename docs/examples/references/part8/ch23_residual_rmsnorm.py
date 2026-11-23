"""Chapter 23: fused residual-add + RMSNorm in Triton.

Run (CPU, Triton interpreter):
    TRITON_INTERPRET=1 python3 ch23_residual_rmsnorm.py
Run (GPU): python3 ch23_residual_rmsnorm.py --bench
"""
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def residual_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, y_ptr, new_res_ptr,
                            n_cols, stride_row, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    off = row * stride_row + cols
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
    h = x + r                                   # residual add, kept in registers
    tl.store(new_res_ptr + off, h.to(new_res_ptr.dtype.element_ty), mask=mask)
    ms = tl.sum(h * h, axis=0) / n_cols         # mean of squares (masked lanes are 0)
    inv = 1.0 / tl.sqrt(ms + eps)               # eps INSIDE the sqrt
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = h * inv * w
    tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty), mask=mask)


def residual_rmsnorm(x, residual, weight, eps=1e-6):
    assert x.is_contiguous() and residual.is_contiguous() and x.shape == residual.shape
    n_rows, n_cols = x.shape
    assert n_cols > 0 and n_cols <= 65536, "one-program-per-row kernel: row must fit one block"
    y = torch.empty_like(x)
    new_res = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    residual_rmsnorm_kernel[(n_rows,)](x, residual, weight, y, new_res,
                                       n_cols, x.stride(0), eps, BLOCK=BLOCK)
    return y, new_res


def reference(x, residual, weight, eps=1e-6):
    h = (x.float() + residual.float())
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return y.to(x.dtype), h.to(x.dtype)


def main():
    torch.manual_seed(0)
    for rows, cols in [(4, 64), (5, 100), (3, 1000)]:   # includes non-power-of-two widths
        x = torch.randn(rows, cols)
        r = torch.randn(rows, cols)
        w = torch.randn(cols)
        y, nr = residual_rmsnorm(x, r, w)
        ry, rnr = reference(x, r, w)
        print(f"{rows}x{cols}: y ok={torch.allclose(y, ry, atol=1e-5)}  "
              f"new_residual ok={torch.allclose(nr, rnr, atol=1e-6)}")
    # eps placement: near-zero row, where eps matters
    x = torch.full((1, 8), 1e-4); r = torch.zeros(1, 8); w = torch.ones(8)
    y, _ = residual_rmsnorm(x, r, w)
    ry, _ = reference(x, r, w)
    print("near-zero row ok:", torch.allclose(y, ry, atol=1e-5))
    # analytic HBM traffic per element-slot (n = rows*cols), eager PyTorch vs fused:
    #   eager: add(r2,w1) + pow(r1,w1) + mean(r1) + mul inv(r1,w1... counted as r1,w1 per op) + mul w(r1,w1)
    #          = 3n + 2n + 1n + 2n + 2n = 10n element transfers
    #   fused: read x,res (2n) + write y,new_res (2n) = 4n   (weight is reused across rows)
    n = 4096 * 8192
    fused, eager = 4 * n * 2, 10 * n * 2          # bf16 = 2 bytes
    print(f"analytic bytes (bf16, 4096x8192): fused={fused/1e6:.0f} MB, eager={eager/1e6:.0f} MB, ratio={eager/fused:.1f}x")


if __name__ == "__main__":
    main()
