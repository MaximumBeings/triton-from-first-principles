"""Chapter 23: fused residual-add + RMSNorm in Triton.

Run (CPU, Triton interpreter):
    TRITON_INTERPRET=1 python3 ch23_residual_rmsnorm.py
    TRITON_INTERPRET=1 python3 ch23_residual_rmsnorm.py --test      (21 self-tests)
    TRITON_INTERPRET=1 python3 ch23_residual_rmsnorm.py --extras    (numerics and traffic tables)
Run (GPU only, UNTESTED on any GPU by the author): python3 ch23_residual_rmsnorm.py --bench
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


# ------------------------------------------------------------- self-tests
def _fp64_ref(x, r, w, eps=1e-6):
    h = x.double() + r.double()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.double(), h


def run_tests():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    torch.manual_seed(1)
    ok_all = True
    for rows in (1, 2, 7):
        for cols in (1, 2, 3, 7, 64, 100, 127, 128, 129, 1000):
            x, r, w = torch.randn(rows, cols), torch.randn(rows, cols), torch.randn(cols)
            y, nr = residual_rmsnorm(x, r, w)
            ry, rh = _fp64_ref(x, r, w)
            ok_all &= bool(torch.allclose(y.double(), ry, atol=2e-5, rtol=2e-5) and torch.allclose(nr.double(), rh, atol=1e-6))
    check("fp32: 30 shapes (rows 1/2/7 x widths 1..1000, incl. every power-of-two neighbour) match the fp64 reference", ok_all)
    x, r, w = torch.randn(3, 100), torch.randn(3, 100), torch.randn(100)
    y, nr = residual_rmsnorm(x, r, w)
    check("output dtypes follow the input dtype", y.dtype == x.dtype and nr.dtype == x.dtype)
    xb, rb, wb = x.bfloat16(), r.bfloat16(), w.bfloat16()
    yb, nrb = residual_rmsnorm(xb, rb, wb)
    yr, hr = reference(xb, rb, wb)
    h32 = xb.float() + rb.float()
    ulp = h32.abs().clamp_min(1e-30).log2().floor().exp2() * 2.0 ** -7      # bf16 spacing at each value
    check("bf16: new residual within 1 bf16 ulp of the fp32 sum (the interpreter truncates, a GPU rounds to nearest; see text)",
          bool(((nrb.float() - h32).abs() <= ulp).all()) and yb.dtype == torch.bfloat16)
    check("bf16: y within 2 bf16 ulps (rel 1.6e-2) of the fp32 reference", bool(torch.allclose(yb.float(), yr.float(), atol=2e-2, rtol=1.6e-2)))
    xh, rh_, wh = x.half(), r.half(), w.half()
    yh, nrh = residual_rmsnorm(xh, rh_, wh)
    check("fp16 inputs work and keep their dtype", yh.dtype == torch.float16 and bool(torch.isfinite(yh).all()))
    # edge values
    y1, _ = residual_rmsnorm(torch.tensor([[3.0]]), torch.tensor([[1.0]]), torch.tensor([2.0]))
    check("width 1: y = h / sqrt(h^2 + eps) * w = 2 (h = 4)", abs(float(y1) - 2.0 * 4.0 / (16.0 + 1e-6) ** 0.5) < 1e-6)
    z, nrz = residual_rmsnorm(torch.zeros(2, 33), torch.zeros(2, 33), torch.ones(33))
    check("all-zero rows give exactly 0, not NaN (eps keeps the division finite)", bool((z == 0).all()) and bool((nrz == 0).all()))
    big = torch.full((1, 64), 1e4)
    yb2, _ = residual_rmsnorm(big, torch.zeros(1, 64), torch.ones(64))
    check("large values (1e4) do not overflow: result is ~1 everywhere", bool(torch.allclose(yb2, torch.ones(1, 64), atol=1e-4)))
    xh2 = torch.full((1, 64), 3e4).half()
    yh2, nrh2 = residual_rmsnorm(xh2, torch.zeros(1, 64).half(), torch.ones(64).half())
    check("fp16 near its max (3e4): fp32 accumulation avoids overflow of the sum of squares", bool(torch.isfinite(yh2).all()) and bool(torch.allclose(yh2.float(), torch.ones(1, 64), atol=1e-2)))
    ya, _ = residual_rmsnorm(x, r, w)
    yb3, _ = residual_rmsnorm(x, r, 2 * w)
    check("linear in the weight: y(2w) == 2 y(w)", bool(torch.allclose(yb3, 2 * ya, atol=1e-6)))
    ys, _ = residual_rmsnorm(10 * x, 10 * r, w)
    check("scale invariance when eps is negligible: y(10x, 10r) ~ y(x, r)", bool(torch.allclose(ys, ya, atol=1e-4)))
    perm = torch.tensor([2, 0, 1])
    yp, _ = residual_rmsnorm(x[perm].contiguous(), r[perm].contiguous(), w)
    check("rows are independent: permuting rows permutes the output", bool(torch.allclose(yp, ya[perm], atol=1e-6)))
    xn = x.clone(); xn[1, 5] = float("nan")
    yn, _ = residual_rmsnorm(xn, r, w)
    check("a NaN poisons only its own row", bool(torch.isnan(yn[1]).all()) and bool(torch.isfinite(yn[0]).all()) and bool(torch.isfinite(yn[2]).all()))
    xc, rc = x.clone(), r.clone()
    residual_rmsnorm(xc, rc, w)
    check("inputs are not modified", bool((xc == x).all()) and bool((rc == r).all()))
    yq, nrq = residual_rmsnorm(x, r, w)
    check("y and new_residual are separate buffers", yq.data_ptr() != nrq.data_ptr())
    far = residual_rmsnorm(torch.full((1, 8), 1e-4), torch.zeros(1, 8), torch.ones(8), eps=1e-6)[0]
    near = residual_rmsnorm(torch.full((1, 8), 1e-4), torch.zeros(1, 8), torch.ones(8), eps=1e-2)[0]
    check("eps matters for a near-zero row (h=1e-4): eps=1e-6 gives 1e-4/sqrt(1.01e-6)=0.0995, eps=1e-2 gives 1e-4/0.1=0.001",
          abs(float(far[0, 0]) - 1e-4 / (1e-8 + 1e-6) ** 0.5) < 1e-5 and abs(float(near[0, 0]) - 1e-4 / (1e-8 + 1e-2) ** 0.5) < 1e-6)
    # documented failures
    def raises(fn):
        try:
            fn(); return False
        except AssertionError:
            return True
    check("a non-contiguous input is rejected (AssertionError)", raises(lambda: residual_rmsnorm(torch.randn(4, 8)[:, ::2], torch.randn(4, 4), torch.ones(4))))
    check("a width above 65536 is rejected", raises(lambda: residual_rmsnorm(torch.zeros(1, 65537), torch.zeros(1, 65537), torch.ones(65537))))
    check("a width of 0 is rejected", raises(lambda: residual_rmsnorm(torch.zeros(1, 0), torch.zeros(1, 0), torch.ones(0))))
    check("shape mismatch between x and residual is rejected", raises(lambda: residual_rmsnorm(torch.zeros(2, 8), torch.zeros(3, 8), torch.ones(8))))
    for name, good in results:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    print(f"{sum(g for _, g in results)}/{len(results)} self-tests pass")
    return all(g for _, g in results)


# ------------------------------------------------------------- extras (numerics and analytic tables; no speed claims)
def extras():
    torch.manual_seed(0)
    print("Experiment A: why accumulate in fp32? Serial sum of squares of a bf16 row, relative error of the mean-square vs fp64")
    print("(torch.sum on bf16 already accumulates in fp32 internally, so a Python loop is used to model a naive bf16 accumulator)")
    print("width   serial bf16 accumulator   fp32 accumulator")
    for n in (256, 1024, 4096):
        e_b = e_f = 0.0
        for _ in range(5):
            h = torch.randn(n).bfloat16()
            sq = (h * h)                                   # each square rounded to bf16, as a bf16 kernel would
            true = (h.double() ** 2).mean()
            acc = torch.zeros((), dtype=torch.bfloat16)
            for v in sq:
                acc = acc + v                              # bf16 + bf16 -> rounded to bf16 every step
            e_b += float(abs(acc.double() / n - true) / true)
            e_f += float(abs((h.float() * h.float()).sum() / n - true) / true)
        print(f"{n:5d}   {e_b/5:21.2e}   {e_f/5:16.2e}")
    print("\nExperiment B: lanes wasted by BLOCK = next_power_of_2(width), for real hidden sizes")
    print("hidden  BLOCK  wasted lanes")
    for n in (2048, 2560, 3584, 4096, 5120, 6144, 7168, 8192, 14336):
        b = triton.next_power_of_2(n)
        print(f"{n:6d}  {b:5d}  {100*(b-n)/b:5.1f} %")
    print("\nExperiment C: bandwidth floor by token count (hidden 8192, bf16, H100 SXM 3.35 TB/s datasheet)")
    print("tokens  fused MB  eager MB  fused floor us  eager floor us")
    for t in (1, 16, 256, 4096):
        n = t * 8192
        f, e = 4 * n * 2, 10 * n * 2
        print(f"{t:6d}  {f/1e6:8.2f}  {e/1e6:8.2f}  {f/3.35e12*1e6:14.2f}  {e/3.35e12*1e6:14.2f}")
    print("(at 1 token the floor is a few hundredths of a microsecond: launch and latency, not bandwidth, decide the real time)")
    print("\nExperiment D: error against fp64 over 20 seeds, fp32 kernel, 8 x 1000")
    worst = 0.0
    for sd in range(20):
        torch.manual_seed(sd)
        x, r, w = torch.randn(8, 1000), torch.randn(8, 1000), torch.randn(1000)
        y, _ = residual_rmsnorm(x, r, w)
        ry, _ = _fp64_ref(x, r, w)
        worst = max(worst, float((y.double() - ry).abs().max()))
    print(f"worst max|error| = {worst:.2e}")


# ------------------------------------------------------------- GPU benchmark (UNTESTED: written without access to a GPU)
def bench():
    if not torch.cuda.is_available():
        print("--bench needs a CUDA GPU; none found. Nothing was measured.")
        return
    from triton.testing import do_bench
    print("hidden  tokens  eager_us  fused_us  speedup  fused_GB/s")
    for hidden in (1024, 4096, 8192, 16384):
        tokens = 4096
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        w = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
        eager = do_bench(lambda: reference(x, r, w), warmup=25, rep=100)
        fused = do_bench(lambda: residual_rmsnorm(x, r, w), warmup=25, rep=100)
        gbs = 4 * tokens * hidden * 2 / (fused * 1e-3) / 1e9
        print(f"{hidden:6d}  {tokens:6d}  {eager*1e3:8.1f}  {fused*1e3:8.1f}  {eager/fused:6.2f}x  {gbs:10.0f}")
    print("Note: `reference` upcasts to fp32 and runs several kernels; compare against your model's real eager path too.")



if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    if "--extras" in sys.argv:
        extras(); sys.exit(0)
    if "--bench" in sys.argv:
        bench(); sys.exit(0)
    main()
