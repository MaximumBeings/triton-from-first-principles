"""Chapter 24: decode attention over a paged KV cache (split-K / flash-decoding style).

Run (CPU, Triton interpreter):
    TRITON_INTERPRET=1 python3 ch24_paged_decode_attention.py
    TRITON_INTERPRET=1 python3 ch24_paged_decode_attention.py --test      (self-tests)
    TRITON_INTERPRET=1 python3 ch24_paged_decode_attention.py --extras    (experiments, analytic tables)
Run (GPU only, UNTESTED by the author): python3 ch24_paged_decode_attention.py --bench
"""
import math
import sys
import torch
import triton
import triton.language as tl

NEG = -1.0e30   # finite "minus infinity": keeps exp(m_old - m_new) well defined for empty splits


@triton.jit
def decode_stage1(q_ptr, k_ptr, v_ptr, bt_ptr, lens_ptr,
                  part_m_ptr, part_l_ptr, part_o_ptr,
                  sm_scale, max_pages, pages_per_split,
                  stride_qb, stride_qh,
                  stride_kp, stride_kt, stride_kh,       # k/v cache: [pages, page_size, kv_heads, D]
                  stride_btb,
                  group,                                  # q heads per kv head (GQA)
                  PAGE: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    kvh = h // group
    seq_len = tl.load(lens_ptr + b)
    d = tl.arange(0, D)
    t = tl.arange(0, PAGE)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d).to(tl.float32)

    m = tl.full((), NEG, tl.float32)
    l = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    first = s * pages_per_split
    for i in range(0, pages_per_split):
        pg = first + i
        tok = pg * PAGE + t                              # logical token positions in this page
        valid = (tok < seq_len) & (tok < max_pages * PAGE)
        page_id = tl.load(bt_ptr + b * stride_btb + tl.minimum(pg, max_pages - 1))   # clamp; `valid` masks the rest
        base = page_id * stride_kp + kvh * stride_kh
        kv_off = base + t[:, None] * stride_kt + d[None, :]
        k = tl.load(k_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(k * q[None, :], axis=1) * sm_scale
        sc = tl.where(valid, sc, NEG)
        m_new = tl.maximum(m, tl.max(sc, axis=0))
        p = tl.where(valid, tl.exp(sc - m_new), 0.0)     # masked lanes contribute exactly 0
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new
    idx = (b * tl.num_programs(1) + h) * NSPLIT + s
    tl.store(part_m_ptr + idx, m)
    tl.store(part_l_ptr + idx, l)
    tl.store(part_o_ptr + idx * D + d, acc)


@triton.jit
def decode_stage2(part_m_ptr, part_l_ptr, part_o_ptr, out_ptr,
                  stride_ob, stride_oh,
                  D: tl.constexpr, NSPLIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    nh = tl.num_programs(1)
    sp = tl.arange(0, NSPLIT)
    d = tl.arange(0, D)
    base = (b * nh + h) * NSPLIT
    m = tl.load(part_m_ptr + base + sp)
    l = tl.load(part_l_ptr + base + sp)
    o = tl.load(part_o_ptr + (base + sp)[:, None] * D + d[None, :])
    m_g = tl.max(m, axis=0)
    w = tl.exp(m - m_g)                                  # empty splits: exp(-1e30 - m_g) == 0
    denom = tl.sum(w * l, axis=0)
    out = tl.sum(w[:, None] * o, axis=0) / denom
    tl.store(out_ptr + b * stride_ob + h * stride_oh + d, out.to(out_ptr.dtype.element_ty))


def paged_decode_attention(q, k_cache, v_cache, block_table, seq_lens, nsplit=4):
    B, H, D = q.shape
    _, PAGE, KVH, _ = k_cache.shape
    assert H % KVH == 0 and (nsplit & (nsplit - 1)) == 0
    group = H // KVH
    max_pages = block_table.shape[1]
    pps = triton.cdiv(max_pages, nsplit)
    pm = torch.empty(B * H * nsplit, dtype=torch.float32)
    pl = torch.empty_like(pm)
    po = torch.empty(B * H * nsplit * D, dtype=torch.float32)
    out = torch.empty_like(q)
    decode_stage1[(B, H, nsplit)](q, k_cache, v_cache, block_table, seq_lens, pm, pl, po,
                                  1.0 / math.sqrt(D), max_pages, pps,
                                  q.stride(0), q.stride(1),
                                  k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                                  block_table.stride(0), group,
                                  PAGE=PAGE, D=D, NSPLIT=nsplit)
    decode_stage2[(B, H)](pm, pl, po, out, out.stride(0), out.stride(1), D=D, NSPLIT=nsplit)
    return out


def reference(q, k_cache, v_cache, block_table, seq_lens):
    B, H, D = q.shape
    _, PAGE, KVH, _ = k_cache.shape
    group = H // KVH
    outs = []
    for b in range(B):
        n = int(seq_lens[b])
        pages = block_table[b, : math.ceil(n / PAGE)].long()
        k = k_cache[pages].reshape(-1, KVH, D)[:n]           # [n, KVH, D]
        v = v_cache[pages].reshape(-1, KVH, D)[:n]
        k = k.repeat_interleave(group, dim=1)                 # [n, H, D]
        v = v.repeat_interleave(group, dim=1)
        sc = torch.einsum("hd,nhd->hn", q[b].float(), k.float()) / math.sqrt(D)
        outs.append(torch.einsum("hn,nhd->hd", sc.softmax(-1), v.float()))
    return torch.stack(outs).to(q.dtype)


def make_case(B=3, H=4, KVH=2, D=16, PAGE=4, lens=(1, 9, 30), n_pages_total=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, D, generator=g)
    kc = torch.randn(n_pages_total, PAGE, KVH, D, generator=g)
    vc = torch.randn(n_pages_total, PAGE, KVH, D, generator=g)
    max_pages = math.ceil(max(lens) / PAGE)
    perm = torch.randperm(n_pages_total, generator=g)         # pages are NOT contiguous or ordered
    bt = torch.zeros(B, max_pages, dtype=torch.int32)
    used = 0
    for b, n in enumerate(lens):
        need = math.ceil(n / PAGE)
        bt[b, :need] = perm[used:used + need].int()
        used += need
    return q, kc, vc, bt, torch.tensor(lens, dtype=torch.int32)


def main():
    q, kc, vc, bt, lens = make_case()
    ref = reference(q, kc, vc, bt, lens)
    for ns in (1, 2, 4, 8):
        out = paged_decode_attention(q, kc, vc, bt, lens, nsplit=ns)
        print(f"nsplit={ns}: matches reference? {torch.allclose(out, ref, atol=1e-5)}  "
              f"max|err|={(out - ref).abs().max().item():.2e}")
    # seq_len == 1 (single token) and a length that is an exact multiple of the page size
    q, kc, vc, bt, lens = make_case(lens=(1, 8, 16), seed=1)
    out = paged_decode_attention(q, kc, vc, bt, lens, nsplit=4)
    print("edge lengths (1, 8, 16) ok?", torch.allclose(out, reference(q, kc, vc, bt, lens), atol=1e-5))
    # KV-cache bytes per token for a Llama-3-70B-shaped model (80 layers, 8 kv heads, D=128, bf16)
    per_tok = 2 * 80 * 8 * 128 * 2
    print(f"KV bytes/token = {per_tok} ({per_tok/1024:.0f} KiB); 128K tokens = {per_tok*131072/2**30:.1f} GiB")


# ------------------------------------------------------------- self-tests
def _ref64(q, kc, vc, bt, lens):
    return reference(q.double(), kc.double(), vc.double(), bt, lens).double()


def _slot_valid(kc, bt, lens):
    """bool [pages, PAGE]: True where a cache slot holds a token of some sequence (a token < its seq_len)."""
    PAGE = kc.shape[1]
    ok = torch.zeros(kc.shape[0], PAGE, dtype=torch.bool)
    for b, n in enumerate(lens.tolist()):
        for pg in range(math.ceil(n / PAGE)):
            cnt = min(PAGE, n - pg * PAGE)
            ok[int(bt[b, pg]), :cnt] = True
    return ok


def run_tests():
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    def raises(fn):
        try:
            fn(); return False
        except AssertionError:
            return True

    # 1-2: a length sweep through page boundaries, every split count
    lens = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33)
    q, kc, vc, bt, ln = make_case(B=len(lens), H=4, KVH=2, D=16, PAGE=4, lens=lens, n_pages_total=120, seed=3)
    ref = _ref64(q, kc, vc, bt, ln)
    outs = {ns: paged_decode_attention(q, kc, vc, bt, ln, nsplit=ns) for ns in (1, 2, 4, 8)}
    check("14 lengths (1..33, on and off page boundaries) x nsplit 1/2/4/8 match the fp64 reference (atol 1e-5)",
          all(torch.allclose(o.double(), ref, atol=1e-5) for o in outs.values()))
    check("the split count does not change the answer: nsplit 1 vs 8 agree to 1e-6",
          bool(torch.allclose(outs[1], outs[8], atol=1e-6)) and bool(torch.allclose(outs[2], outs[4], atol=1e-6)))
    # 3: relabel physical pages
    P = torch.randperm(kc.shape[0], generator=torch.Generator().manual_seed(5))
    kc2, vc2 = torch.empty_like(kc), torch.empty_like(vc)
    kc2[P], vc2[P] = kc, vc
    out_p = paged_decode_attention(q, kc2, vc2, P[bt.long()].int(), ln, nsplit=4)
    check("relabelling every physical page (cache permuted, block table remapped) leaves the output bit-identical",
          torch.equal(out_p, outs[4]))
    # 4: garbage in every slot that is not a live token
    ok = _slot_valid(kc, bt, ln)
    kg, vg = kc.clone(), vc.clone()
    kg[~ok] = float("nan"); vg[~ok] = float("nan")
    out_g = paged_decode_attention(q, kg, vg, bt, ln, nsplit=4)
    check("NaN in every slot beyond a sequence's length (partial last pages and unowned pages) does not leak",
          bool(torch.isfinite(out_g).all()) and bool(torch.allclose(out_g, outs[4], atol=1e-6)))
    # 5: garbage in unused block-table entries
    btg = bt.clone()
    for b, n in enumerate(ln.tolist()):
        btg[b, math.ceil(n / 4):] = 10 ** 6
    out_b = paged_decode_attention(q, kc, vc, btg, ln, nsplit=4)
    check("unused block-table entries holding an out-of-range page id (10^6) are never dereferenced", torch.allclose(out_b, outs[4], atol=1e-6))
    # 6: dims sweep
    ok6 = True
    for H, KVH, D, PAGE in [(4, 4, 16, 4), (8, 2, 32, 8), (4, 1, 16, 16), (2, 2, 64, 4)]:
        q6, k6, v6, b6, l6 = make_case(B=2, H=H, KVH=KVH, D=D, PAGE=PAGE, lens=(3, 17), n_pages_total=20, seed=7)
        ok6 &= bool(torch.allclose(paged_decode_attention(q6, k6, v6, b6, l6, nsplit=2).double(), _ref64(q6, k6, v6, b6, l6), atol=1e-5))
    check("MHA (group 1), GQA (group 4) and MQA (1 kv head), D = 16/32/64, page = 4/8/16 all match", ok6)
    # 7: single token
    q7, k7, v7, b7, l7 = make_case(B=2, H=4, KVH=2, D=16, PAGE=4, lens=(1, 1), n_pages_total=10, seed=2)
    o7 = paged_decode_attention(q7, k7, v7, b7, l7, nsplit=4)
    want = torch.stack([torch.stack([v7[int(b7[b, 0]), 0, h // 2] for h in range(4)]) for b in range(2)])
    check("one token: softmax of one score is 1, so the output is exactly that token's V", torch.allclose(o7, want, atol=1e-6))
    # 8: uniform keys give the mean of V
    q8, k8, v8, b8, l8 = make_case(B=1, H=2, KVH=1, D=16, PAGE=4, lens=(11,), n_pages_total=10, seed=4)
    k8 = k8[0:1, 0:1].expand_as(k8).clone()
    o8 = paged_decode_attention(q8, k8, v8, b8, l8, nsplit=4)
    toks = torch.cat([v8[int(b8[0, pg])][: min(4, 11 - 4 * pg)] for pg in range(3)])      # [11, 1, 16]
    check("identical keys: every weight is 1/11, so the output is the mean of the 11 V rows (hand computation, not the reference)",
          torch.allclose(o8[0, 0], toks[:, 0].mean(0), atol=1e-6) and torch.allclose(o8[0, 1], toks[:, 0].mean(0), atol=1e-6))
    # 9: one dominant key
    q9, k9, v9, b9, l9 = make_case(B=1, H=1, KVH=1, D=16, PAGE=4, lens=(10,), n_pages_total=10, seed=6)
    pg, sl = int(b9[0, 1]), 2                                  # logical token 6
    k9[pg, sl, 0] = q9[0, 0] * (100.0 * 4.0 / float(q9[0, 0] @ q9[0, 0]))   # score = 100 after the 1/sqrt(16) scaling
    o9 = paged_decode_attention(q9, k9, v9, b9, l9, nsplit=4)
    check("one key with a score 100 above the rest takes all the weight: output equals that token's V", torch.allclose(o9[0, 0], v9[pg, sl, 0], atol=1e-5))
    # 10: huge scores
    o10 = paged_decode_attention(q * 200, kc, vc, bt, ln, nsplit=4)
    r10 = _ref64(q * 200, kc, vc, bt, ln)
    check("scores in the hundreds (q x 200) do not overflow exp: finite and equal to the reference", bool(torch.isfinite(o10).all()) and torch.allclose(o10.double(), r10, atol=1e-4))
    # 11: convex hull
    hull = True
    for b, n in enumerate(ln.tolist()):
        vv = torch.cat([vc[int(bt[b, pg])][: min(4, n - 4 * pg)] for pg in range(math.ceil(n / 4))])   # [n, KVH, D]
        for h in range(4):
            x = vv[:, h // 2]
            hull &= bool(((outs[4][b, h] >= x.min(0).values - 1e-6) & (outs[4][b, h] <= x.max(0).values + 1e-6)).all())
    check("every output coordinate lies between the min and max of that sequence's V (a softmax-weighted average)", hull)
    # 12: reversing full pages
    ql, kl, vl, bl, ll = make_case(B=3, H=4, KVH=2, D=16, PAGE=4, lens=(8, 16, 32), n_pages_total=30, seed=8)
    o_a = paged_decode_attention(ql, kl, vl, bl, ll, nsplit=4)
    br = bl.clone()
    for b, n in enumerate(ll.tolist()):
        br[b, : n // 4] = bl[b, : n // 4].flip(0)
    check("reversing the page order of full sequences leaves the output unchanged: position lives in K (RoPE), not in the cache layout",
          torch.allclose(paged_decode_attention(ql, kl, vl, br, ll, nsplit=4), o_a, atol=1e-5))
    # 13: batch independence
    qb, kb, vb, bb, lb = make_case(B=3, H=4, KVH=2, D=16, PAGE=4, lens=(5, 9, 13), n_pages_total=30, seed=9)
    o_b0 = paged_decode_attention(qb, kb, vb, bb, lb, nsplit=4)
    kb2, vb2 = kb.clone(), vb.clone()
    for pgi in bb[0, :2].tolist():
        kb2[pgi] += 10.0; vb2[pgi] -= 10.0
    o_b1 = paged_decode_attention(qb, kb2, vb2, bb, lb, nsplit=4)
    check("rewriting sequence 0's pages leaves sequences 1 and 2 bit-identical (and changes sequence 0)",
          torch.equal(o_b0[1:], o_b1[1:]) and not torch.allclose(o_b0[0], o_b1[0], atol=1e-3))
    # 14: bf16
    qh, kh, vh = qb.bfloat16(), kb.bfloat16(), vb.bfloat16()
    o_h = paged_decode_attention(qh, kh, vh, bb, lb, nsplit=4)
    check("bf16 inputs: output is bf16 and within 2e-2 of the fp32 reference",
          o_h.dtype == torch.bfloat16 and torch.allclose(o_h.float(), reference(qh, kh, vh, bb, lb).float(), atol=2e-2))
    # 15: determinism, inputs untouched
    snap = [t.clone() for t in (qb, kb, vb, bb, lb)]
    o_r = paged_decode_attention(qb, kb, vb, bb, lb, nsplit=4)
    check("two runs are bit-identical", torch.equal(o_r, o_b0))
    check("q, K, V, block table and lengths are not modified", all(torch.equal(a, b_) for a, b_ in zip(snap, (qb, kb, vb, bb, lb))))
    # 17: length beyond capacity
    qc, kc_, vc_, bc, lc = make_case(B=1, H=2, KVH=1, D=16, PAGE=4, lens=(12,), n_pages_total=10, seed=10)
    o_cap = paged_decode_attention(qc, kc_, vc_, bc, torch.tensor([40], dtype=torch.int32), nsplit=2)
    o_ok = paged_decode_attention(qc, kc_, vc_, bc, lc, nsplit=2)
    check("seq_len 40 with a 3-page table attends to the 12 tokens the table covers (same as seq_len 12): silently clamped, see text", torch.allclose(o_cap, o_ok, atol=1e-6))
    # 18: empty sequence
    o_e = paged_decode_attention(qc, kc_, vc_, bc, torch.tensor([0], dtype=torch.int32), nsplit=2)
    check("seq_len 0 returns NaN (0/0 in stage 2): documented limitation, the scheduler must not launch empty sequences", bool(torch.isnan(o_e).all()))
    # 19: split count and heads
    check("nsplit not a power of two is rejected", raises(lambda: paged_decode_attention(qc, kc_, vc_, bc, lc, nsplit=3)))
    check("query heads not divisible by kv heads is rejected", raises(lambda: paged_decode_attention(torch.randn(3, 5, 16), kb, vb, bb, lb, nsplit=2)))
    for name, good in results:
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
    print(f"{sum(g for _, g in results)}/{len(results)} self-tests pass")
    return all(g for _, g in results)


# ------------------------------------------------------------- extras
def extras():
    print("Experiment A: programs launched by stage 1 = batch x q_heads x nsplit (64 q heads; an H100 SXM has 132 SMs, datasheet)")
    print("batch  nsplit=1   2    4    8   16   32")
    for B in (1, 4, 16, 64):
        print(f"{B:5d}  " + "  ".join(f"{B*64*ns:5d}" for ns in (1, 2, 4, 8, 16, 32)))
    print("any entry below ~132 leaves SMs idle; a single sequence (batch 1) with 64 heads already gives 64 programs, so it needs nsplit >= 4 to pass 132")
    print("\nExperiment B: error against fp64 vs context length (B=1, H=2, KVH=1, D=16, PAGE=16, fp32 kernel)")
    print("tokens  nsplit=1     nsplit=8")
    for n in (64, 256, 1024):
        q, kc, vc, bt, ln = make_case(B=1, H=2, KVH=1, D=16, PAGE=16, lens=(n,), n_pages_total=n // 16 + 4, seed=n)
        r = _ref64(q, kc, vc, bt, ln)
        e1 = float((paged_decode_attention(q, kc, vc, bt, ln, nsplit=1).double() - r).abs().max())
        e8 = float((paged_decode_attention(q, kc, vc, bt, ln, nsplit=8).double() - r).abs().max())
        print(f"{n:6d}  {e1:9.2e}  {e8:9.2e}")
    print("\nExperiment C: wasted page iterations in a mixed batch (the grid is sized by the LONGEST sequence), PAGE=16, nsplit=4")
    mixes = {"uniform 2000 x 8": [2000] * 8, "long tail": [4000, 2000, 500, 100, 60, 30, 10, 1], "one long + 7 short": [8000] + [50] * 7}
    print("batch                 live pages  iterations  fully-masked share")
    for name, ls in mixes.items():
        maxp = math.ceil(max(ls) / 16)
        pps = math.ceil(maxp / 4)
        its = len(ls) * 4 * pps
        live = sum(math.ceil(n / 16) for n in ls)
        print(f"{name:20s}  {live:10d}  {its:10d}  {100*(1-live/its):15.1f} %")
    print("masked iterations load nothing (the loads are masked) but still cost loop and issue time; per-sequence split counts or sorting by length reduce it")
    print("\nExperiment D: KV bytes per token and per decode step (bf16; model shapes are ASSUMED round numbers, check your model's config)")
    print("model shape                          B/token   per-step read at 8K   at 128K   floor at 128K (3.35 TB/s)")
    for name, layers, kvh, d in (("70B-like: 80 layers, 8 kv heads, D=128", 80, 8, 128), ("8B-like: 32 layers, 8 kv heads, D=128", 32, 8, 128), ("no GQA: 32 layers, 32 kv heads, D=128", 32, 32, 128)):
        per = 2 * layers * kvh * d * 2
        print(f"{name:38s} {per/1024:5.0f} KiB   {per*8192/2**30:10.2f} GiB  {per*131072/2**30:7.1f} GiB  {per*131072/3.35e12*1e3:8.1f} ms")
    print("every decode step of ONE sequence reads its whole cache once, so the floor per generated token is bytes / bandwidth")


# ------------------------------------------------------------- GPU benchmark (UNTESTED: written without access to a GPU)
def bench():
    if not torch.cuda.is_available():
        print("--bench needs a CUDA GPU; none found. Nothing was measured.")
        return
    from triton.testing import do_bench
    B, H, KVH, D, PAGE, n = 4, 32, 8, 128, 16, 8192
    q, kc, vc, bt, ln = make_case(B=B, H=H, KVH=KVH, D=D, PAGE=PAGE, lens=(n,) * B, n_pages_total=B * n // PAGE + 8)
    q, kc, vc, bt, ln = (t.cuda() for t in (q, kc, vc, bt, ln))
    q, kc, vc = q.bfloat16(), kc.bfloat16(), vc.bfloat16()
    kv_bytes = 2 * B * n * KVH * D * 2
    print("nsplit  time_us  achieved_GB/s   (bytes = K and V once)")
    for ns in (1, 2, 4, 8, 16, 32):
        t = do_bench(lambda: paged_decode_attention(q, kc, vc, bt, ln, nsplit=ns), warmup=25, rep=100)
        print(f"{ns:6d}  {t*1e3:7.1f}  {kv_bytes/(t*1e-3)/1e9:12.0f}")
    print("NOTE: the wrapper allocates its scratch with torch.empty on the CPU; move those tensors to q.device before running on a GPU.")



if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    if "--extras" in sys.argv:
        extras(); sys.exit(0)
    if "--bench" in sys.argv:
        bench(); sys.exit(0)
    main()
