"""Chapter 25: weight-only INT4 (and simulated FP4) quantization with an in-kernel dequant GEMM.

Run (CPU, Triton interpreter):
    TRITON_INTERPRET=1 python3 ch25_int4_weight_only.py
    TRITON_INTERPRET=1 python3 ch25_int4_weight_only.py --test      (self-tests)
    TRITON_INTERPRET=1 python3 ch25_int4_weight_only.py --extras    (experiments)
Run (GPU only, UNTESTED by the author): python3 ch25_int4_weight_only.py --bench
Everything here is numerics only. No latency or VRAM-bandwidth claim is made.
"""
import sys
import torch
import triton
import triton.language as tl

GROUP = 32  # quantization group size along K


# ---------------------------------------------------------------- quantization (host side)
def quantize_int4(w, group=GROUP, clip_search=False):
    """w: [N, K] float. Returns packed uint8 [N, K//2], scales [N, K//group] (symmetric, range -8..7)."""
    N, K = w.shape
    assert K % group == 0 and K % 2 == 0
    wg = w.reshape(N, K // group, group)
    amax = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    best_scale = amax / 7.0
    if clip_search:                                  # calibration: pick the clip ratio minimizing MSE
        best_err = torch.full_like(amax, float("inf"))
        for ratio in torch.linspace(0.6, 1.0, 17):
            sc = amax * ratio / 7.0
            q = torch.round(wg / sc).clamp(-8, 7)
            err = ((q * sc - wg) ** 2).sum(-1, keepdim=True)
            better = err < best_err
            best_scale = torch.where(better, sc, best_scale)
            best_err = torch.where(better, err, best_err)
    q = torch.round(wg / best_scale).clamp(-8, 7).to(torch.int8)
    q = (q + 8).to(torch.uint8).reshape(N, K)         # offset to 0..15
    packed = q[:, 0::2] | (q[:, 1::2] << 4)           # low nibble = even k, high nibble = odd k
    return packed.contiguous(), best_scale.reshape(N, K // group)


def dequantize_int4(packed, scales, group=GROUP):
    N, Kh = packed.shape
    lo = (packed & 0xF).to(torch.float32) - 8.0
    hi = (packed >> 4).to(torch.float32) - 8.0
    q = torch.stack([lo, hi], dim=-1).reshape(N, Kh * 2)
    return (q.reshape(N, -1, group) * scales[..., None]).reshape(N, Kh * 2)


FP4_E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])   # magnitudes representable by E2M1


def fake_quant_fp4(w, block=16):
    """Simulated FP4 (E2M1 values, per-block scale). Hopper has NO FP4 tensor cores: this is numerics only."""
    N, K = w.shape
    wb = w.reshape(N, K // block, block)
    scale = (wb.abs().amax(-1, keepdim=True) / 6.0).clamp_min(1e-8)
    x = (wb / scale).abs()
    idx = (x[..., None] - FP4_E2M1).abs().argmin(-1)
    return (FP4_E2M1[idx] * torch.sign(wb) * scale).reshape(N, K)


# ---------------------------------------------------------------- GPU kernel: dequant inside the GEMM
@triton.jit
def w4a16_kernel(a_ptr, wp_ptr, s_ptr, c_ptr, M, N, K,
                 stride_am, stride_wn, stride_sn, stride_cm,
                 GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)                         # logical k indices
        a = tl.load(a_ptr + rm[:, None] * stride_am + ks[None, :],
                    mask=(rm[:, None] < M) & (ks[None, :] < K), other=0.0).to(tl.float32)
        byte = tl.load(wp_ptr + rn[None, :] * stride_wn + (ks[:, None] // 2),
                       mask=(rn[None, :] < N) & (ks[:, None] < K), other=0)
        nib = tl.where((ks[:, None] % 2) == 0, byte & 0xF, byte >> 4)   # pick low or high nibble
        sc = tl.load(s_ptr + rn[None, :] * stride_sn + (ks[:, None] // GROUP),
                     mask=(rn[None, :] < N) & (ks[:, None] < K), other=0.0)
        w = (nib.to(tl.float32) - 8.0) * sc                       # dequantize in registers
        acc += tl.dot(a, w, input_precision="ieee")
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def w4a16_matmul(a, packed, scales, K, group=GROUP):
    M = a.shape[0]
    N = packed.shape[0]
    c = torch.empty(M, N, dtype=torch.float32)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    w4a16_kernel[grid](a, packed, scales, c, M, N, K,
                       a.stride(0), packed.stride(0), scales.stride(0), c.stride(0),
                       GROUP=group, BLOCK_M=16, BLOCK_N=16, BLOCK_K=32)
    return c


def main():
    torch.manual_seed(0)
    N, K = 48, 128
    # heavy-tailed weights make clipping matter (a few outliers per group)
    w = torch.randn(N, K) * (1 + 4 * (torch.rand(N, K) > 0.98))
    x = torch.randn(5, K)

    packed, sc = quantize_int4(w)
    packed_c, sc_c = quantize_int4(w, clip_search=True)
    wd, wd_c = dequantize_int4(packed, sc), dequantize_int4(packed_c, sc_c)
    print(f"packed bytes: {packed.numel()} vs fp16 bytes: {w.numel() * 2}  "
          f"(+ scales {sc.numel() * 2} B at fp16) -> {(packed.numel() + sc.numel() * 2) / (w.numel() * 2):.2f}x of fp16")

    def rel(a, b): return ((a - b).norm() / b.norm()).item()
    print(f"weight rel-error  RTN={rel(wd, w):.4f}  clip-search={rel(wd_c, w):.4f}  "
          f"clip better? {rel(wd_c, w) < rel(wd, w)}")
    p16, s16 = quantize_int4(w, group=16, clip_search=True)
    print(f"same block size 16: INT4(clip-search)={rel(dequantize_int4(p16, s16, 16), w):.4f}  "
          f"simulated FP4 (E2M1)={rel(fake_quant_fp4(w, 16), w):.4f}  (numerics only, not speed)")

    # the kernel must equal x @ dequant(W).T exactly (same math), independent of quantization error
    ref = x @ wd_c.T
    out = w4a16_matmul(x, packed_c, sc_c, K)
    print(f"kernel matches dequantized reference? {torch.allclose(out, ref, atol=1e-4)}  "
          f"max|err|={(out - ref).abs().max().item():.2e}")
    print(f"output rel-error vs full-precision x@W.T = {rel(out, x @ w.T):.4f}")
    # odd M, N not multiples of the tile
    out2 = w4a16_matmul(x[:3], packed_c[:37], sc_c[:37], K)
    print("ragged shapes ok?", torch.allclose(out2, x[:3] @ wd_c[:37].T, atol=1e-4))


# ------------------------------------------------------------- self-tests
def _codes(packed):
    """integer codes -8..7 recovered from packed bytes, [N, K]"""
    lo = (packed & 0xF).to(torch.int32) - 8
    hi = (packed >> 4).to(torch.int32) - 8
    return torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def run_tests():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    def raises(fn):
        try:
            fn(); return False
        except AssertionError:
            return True

    torch.manual_seed(0)
    N, K = 48, 128
    w = torch.randn(N, K) * (1 + 4 * (torch.rand(N, K) > 0.98))
    packed, sc = quantize_int4(w)
    packed_c, sc_c = quantize_int4(w, clip_search=True)
    wd, wd_c = dequantize_int4(packed, sc), dequantize_int4(packed_c, sc_c)

    # --- host side: packing and quantizer
    hand = torch.tensor([[0xA3, 0x08]], dtype=torch.uint8)                 # 0xA3: low nibble 3 -> -5, high 10 -> +2 ; 0x08: low 8 -> 0, high 0 -> -8
    check("hand case: byte 0xA3 is [-5, +2] and byte 0x08 is [0, -8] (low nibble = even k)",
          _codes(hand).tolist() == [[-5, 2, 0, -8]] and torch.equal(dequantize_int4(hand, torch.ones(1, 1), 4), torch.tensor([[-5.0, 2.0, 0.0, -8.0]])))
    check("packed is uint8 [N, K/2]; scales are [N, K/group]", packed.dtype == torch.uint8 and packed.shape == (N, K // 2) and sc.shape == (N, K // GROUP))
    ints = torch.randint(-7, 8, (N, K)).float()
    ints[:, ::GROUP] = 7.0                                                # force amax = 7 so the scale is exactly 1
    pi, si = quantize_int4(ints)
    check("weights already on the grid (integers -7..7, amax 7 in every group) round-trip exactly", torch.equal(dequantize_int4(pi, si), ints) and torch.allclose(si, torch.ones_like(si)))
    wg, dg = w.reshape(N, -1, GROUP), wd.reshape(N, -1, GROUP)
    bound = (sc.reshape(N, -1, 1) / 2) * (1 + 1e-5)
    check("RTN error is at most half a scale step for every element (no clipping happens at amax/7)", bool(((wg - dg).abs() <= bound).all()))
    check("RTN never uses code -8 (the symmetric range wastes one of the 16 codes)", int(_codes(packed).min()) >= -7)
    check("clip-search does use -8 on this data", int(_codes(packed_c).min()) == -8)
    z, zs = quantize_int4(torch.zeros(4, 64))
    check("all-zero weights come back as exact zeros, no NaN", torch.equal(dequantize_int4(z, zs), torch.zeros(4, 64)))
    cst = torch.full((2, 64), -0.37)
    pc, scc = quantize_int4(cst)
    check("a constant group is reproduced (to float32 precision)", torch.allclose(dequantize_int4(pc, scc), cst, atol=1e-6))
    e_r = ((wd - w).reshape(N, -1, GROUP) ** 2).sum(-1)
    e_c = ((wd_c - w).reshape(N, -1, GROUP) ** 2).sum(-1)
    check("clip-search is no worse than RTN in EVERY group (ratio 1.0 is in its search set)", bool((e_c <= e_r + 1e-9).all()) and float(e_c.sum()) < float(e_r.sum()))
    w2 = w.clone(); w2[3, 40:70] *= 9.0
    p2, s2 = quantize_int4(w2)
    d2 = dequantize_int4(p2, s2)
    other = torch.ones(N, K, dtype=torch.bool); other[3, 32:96] = False
    check("groups are independent: changing weights in two groups of row 3 changes no other group's result", torch.equal(d2[other], wd[other]))
    pd, sd = quantize_int4(2 * w)
    check("RTN is scale-equivariant: quantize(2w) dequantizes to 2 x dequantize(w)", torch.allclose(dequantize_int4(pd, sd), 2 * wd, atol=1e-6))
    pn, sn = quantize_int4(-w)
    check("RTN is sign-symmetric: quantize(-w) dequantizes to -dequantize(w)", torch.allclose(dequantize_int4(pn, sn), -wd, atol=1e-6))
    s16 = sc.half().float()
    check("fp16 scales (what the 0.28x byte count assumes) change the dequantized weights by at most 2^-10 relative",
          bool(((dequantize_int4(packed, s16) - wd).abs() <= wd.abs() * 2 ** -10 + 1e-12).all()))
    check("bytes: INT4 + fp16 scales is exactly 0.28125 of fp16 for group 32 (0.25 + 1/32)",
          abs((packed.numel() + sc.numel() * 2) / (w.numel() * 2) - 0.28125) < 1e-12)
    check("K not a multiple of the group, or odd K, is rejected", raises(lambda: quantize_int4(torch.randn(2, 50))) and raises(lambda: quantize_int4(torch.randn(2, 33), group=1)))

    # --- FP4 simulation
    f = fake_quant_fp4(w, 16)
    scl = (w.reshape(N, -1, 16).abs().amax(-1, keepdim=True) / 6.0).clamp_min(1e-8)
    mags = (f.reshape(N, -1, 16).abs() / scl)
    on_grid = (mags[..., None] - FP4_E2M1).abs().min(-1).values < 1e-4
    check("every FP4 output is scale x one of {0, .5, 1, 1.5, 2, 3, 4, 6} (up to float32 rounding)", bool(on_grid.all()))
    check("FP4 quantization is idempotent", torch.allclose(fake_quant_fp4(f, 16), f, atol=1e-6))
    blk = torch.zeros(1, 16); blk[0, :8] = torch.tensor([6.0, 0.2, 0.3, 2.4, 2.6, 5.1, -4.9, -0.2])
    check("hand case (block max 6, scale 1): [6, .2, .3, 2.4, 2.6, 5.1, -4.9, -.2] -> [6, 0, .5, 2, 3, 6, -4, 0]",
          fake_quant_fp4(blk, 16)[0, :8].tolist() == [6.0, 0.0, 0.5, 2.0, 3.0, 6.0, -4.0, 0.0])
    check("FP4 of zeros is zeros", torch.equal(fake_quant_fp4(torch.zeros(2, 32), 16), torch.zeros(2, 32)))

    # --- kernel
    x = torch.randn(5, K)
    out = w4a16_matmul(x, packed_c, sc_c, K)
    check("kernel equals x @ dequant(W).T (atol 1e-4)", torch.allclose(out, x @ wd_c.T, atol=1e-4))
    ok = True
    for M, Nn, Kk, g in [(1, 1, 32, 32), (17, 16, 64, 32), (3, 37, 96, 32), (16, 48, 128, 16), (2, 20, 128, 64), (5, 20, 48, 16), (4, 18, 80, 16)]:
        wq = torch.randn(Nn, Kk); xq = torch.randn(M, Kk)
        pq, sq = quantize_int4(wq, group=g, clip_search=True)
        ok &= bool(torch.allclose(w4a16_matmul(xq, pq, sq, Kk, group=g), xq @ dequantize_int4(pq, sq, g).T, atol=1e-4))
    check("ragged M and N, K = 32..128 including K = 48 and 80 (not a multiple of the 32-wide K tile), groups 16/32/64 all match", ok)
    wh = torch.zeros(16, 64); wh[:, 0] = 1.0                              # only k=0 (a LOW nibble) is nonzero
    ph, sh = quantize_int4(wh)
    xh = torch.arange(64, dtype=torch.float32).repeat(2, 1) + 1
    check("hand case, nibble order: only k=0 non-zero, so the output is x[:, 0] (a swapped nibble reads k=1)",
          torch.allclose(w4a16_matmul(xh, ph, sh, 64), torch.ones(2, 16) * xh[:, :1], atol=1e-5))
    wh2 = torch.zeros(16, 64); wh2[:, 1] = 1.0                            # only k=1 (a HIGH nibble) is nonzero
    ph2, sh2 = quantize_int4(wh2)
    check("hand case, nibble order: only k=1 non-zero, so the output is x[:, 1]", torch.allclose(w4a16_matmul(xh, ph2, sh2, 64), torch.ones(2, 16) * xh[:, 1:2], atol=1e-5))
    pk = torch.full((16, 32), 9 | (9 << 4), dtype=torch.uint8)            # every code is +1
    sk = torch.tensor([1.0, 10.0]).repeat(16, 1)                          # group 0 scale 1, group 1 scale 10
    ones = torch.ones(2, 64)
    check("hand case, scale per group: all codes +1, scales [1, 10] -> sum = 32 x 1 + 32 x 10 = 352", torch.allclose(w4a16_matmul(ones, pk, sk, 64), torch.full((2, 16), 352.0), atol=1e-3))
    wk = torch.randn(20, 48); xk = torch.randn(4, 48)
    pk2, sk2 = quantize_int4(wk, group=16, clip_search=True)
    pad = torch.full((20, 3 + 2), float("nan")); pad[:, :3] = sk2          # the 2 padding columns are exactly what an unmasked read past K would hit
    check("K = 48 with a 32-wide K tile: NaN scales stored just past the real K are never used (the K mask on the scale load)",
          bool(torch.isfinite(w4a16_matmul(xk, pk2, pad[:, :3], 48, group=16)).all()))
    xpad = torch.full((4, 48 + 2), float("nan")); xpad[:, :48] = xk       # NaN just past K: an unmasked activation load would read it
    check("K = 48: NaN activations stored just past the real K are never used (the K mask on the activation load)",
          bool(torch.isfinite(w4a16_matmul(xpad[:, :48], pk2, sk2, 48, group=16)).all()))
    xb = x.bfloat16()
    check("bf16 activations are accepted and match the float reference to 5e-2", torch.allclose(w4a16_matmul(xb, packed_c, sc_c, K), xb.float() @ wd_c.T, atol=5e-2))
    snap = (packed_c.clone(), sc_c.clone(), x.clone())
    o1, o2 = w4a16_matmul(x, packed_c, sc_c, K), w4a16_matmul(x, packed_c, sc_c, K)
    check("kernel is deterministic and does not modify its inputs", torch.equal(o1, o2) and torch.equal(snap[0], packed_c) and torch.equal(snap[1], sc_c) and torch.equal(snap[2], x))
    for name, good in results:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    print(f"{sum(g for _, g in results)}/{len(results)} self-tests pass")
    return all(g for _, g in results)


# ------------------------------------------------------------- extras
def extras():
    torch.manual_seed(0)
    N, K = 64, 512
    def heavy(frac):
        return torch.randn(N, K) * (1 + 4 * (torch.rand(N, K) < frac))
    w = heavy(0.02)
    print("Experiment A: group size vs weight error and bytes (2% outliers, N=64, K=512, fp16 scales assumed in the byte count)")
    print("group   RTN err   clip err   bytes vs fp16")
    for g in (16, 32, 64, 128, 512):
        p, s_ = quantize_int4(w, g); pc, sc_ = quantize_int4(w, g, clip_search=True)
        print(f"{g:5d}   {_rel(dequantize_int4(p, s_, g), w):.4f}    {_rel(dequantize_int4(pc, sc_, g), w):.4f}     {(p.numel() + s_.numel()*2)/(w.numel()*2):.4f}")
    print("\nExperiment B: outlier fraction vs weight error, group 32")
    print("outliers   RTN err   clip err   clip gain")
    for fr in (0.0, 0.005, 0.02, 0.05):
        wf = heavy(fr)
        r = _rel(dequantize_int4(*quantize_int4(wf)), wf); c = _rel(dequantize_int4(*quantize_int4(wf, clip_search=True)), wf)
        print(f"{100*fr:6.1f} %   {r:.4f}    {c:.4f}    {100*(r-c)/r:5.1f} %")
    print("\nExperiment C: error of the layer OUTPUT x @ W.T against weight error (group 32, clip search, 256 random activations)")
    x = torch.randn(256, K)
    p, s_ = quantize_int4(w, clip_search=True)
    wd = dequantize_int4(p, s_)
    print(f"weight rel-error {_rel(wd, w):.4f}   output rel-error {_rel(x @ wd.T, x @ w.T):.4f}   (random Gaussian x only: this does not test real activations, whose outlier channels can behave differently)")
    print("\nExperiment D: scale precision, group 32 RTN")
    p, s_ = quantize_int4(w)
    for name, sc2 in (("fp32", s_), ("fp16", s_.half().float()), ("bf16", s_.bfloat16().float())):
        print(f"{name}: weight rel-error {_rel(dequantize_int4(p, sc2), w):.5f}")
    print("\nExperiment E: code usage (share of weights at each end of the range), group 32")
    p, _ = quantize_int4(w); pc, _ = quantize_int4(w, clip_search=True)
    for name, pp in (("RTN", p), ("clip-search", pc)):
        c = _codes(pp)
        print(f"{name:12s} code -8: {100*float((c == -8).float().mean()):.3f} %   code +7: {100*float((c == 7).float().mean()):.3f} %   code -7: {100*float((c == -7).float().mean()):.3f} %")
    print("\nExperiment F: decode bandwidth floor for the WEIGHTS of a 70e9-parameter model (assumed count; ignores embeddings, KV cache, activations)")
    for name, bpp in (("fp16", 2.0), ("INT4 group 32 + fp16 scales", 0.5 + 2 / 32), ("INT4 group 128 + fp16 scales", 0.5 + 2 / 128)):
        print(f"{name:30s} {70e9*bpp/1e9:6.1f} GB   floor {70e9*bpp/3.35e12*1e3:5.1f} ms/token at 3.35 TB/s (datasheet)")
    print("batch-1 decode reads every weight once per token; at large batch or in prefill the GEMM is compute-bound and INT4 helps little")


# ------------------------------------------------------------- GPU benchmark (UNTESTED: written without access to a GPU)
def bench():
    if not torch.cuda.is_available():
        print("--bench needs a CUDA GPU; none found. Nothing was measured.")
        return
    from triton.testing import do_bench
    N, K = 8192, 8192
    w = torch.randn(N, K)
    packed, sc = quantize_int4(w, clip_search=True)
    print("NOTE: w4a16_matmul allocates its output on the CPU and uses a fixed 16 x 16 x 32 tile: fix both before trusting any number from this function.")
    print("M    fp16_torch_us   int4_triton_us")
    wf = w.cuda().half()
    for M in (1, 8, 32, 128):
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        t_f = do_bench(lambda: x @ wf.T, warmup=25, rep=100)
        t_q = do_bench(lambda: w4a16_matmul(x, packed.cuda(), sc.cuda(), K), warmup=25, rep=100)
        print(f"{M:3d}  {t_f*1e3:12.1f}  {t_q*1e3:14.1f}")



if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    if "--extras" in sys.argv:
        extras(); sys.exit(0)
    if "--bench" in sys.argv:
        bench(); sys.exit(0)
    main()
