"""Chapter 24: decode attention over a paged KV cache (split-K / flash-decoding style).

Run (CPU, Triton interpreter):
    TRITON_INTERPRET=1 python3 ch24_paged_decode_attention.py
"""
import math
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


if __name__ == "__main__":
    main()
