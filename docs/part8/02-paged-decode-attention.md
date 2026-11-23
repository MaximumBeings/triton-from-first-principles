# Chapter 24: Decode Attention Over a Paged KV Cache

> Training attention reads a whole sequence at once. Serving attention reads the entire history to produce a single token. One query, thousands of keys, and a cache stored in scattered fixed-size pages: the kernel is small, and almost everything interesting about it is about memory.

!!! info "What was and was not validated"
    The kernels and outputs below ran in the Triton **interpreter** (`TRITON_INTERPRET=1`, CPU) against a reference. That validates numerics, masking, paging and the split-K combine. **No H100 timing is claimed.** The split-count tuning guidance is a procedure to run, not a result.

!!! info "Supports Projects 3 and 7"
    Offloading the KV cache (Project 3) and speculative decoding (Project 7) both need a fast decode-attention kernel. This chapter builds the reference one.

## What you will understand by the end of this chapter

- Why decode attention is memory-bound and why a single program per head cannot fill a GPU on long contexts
- How a paged KV cache maps logical token positions to physical pages through a block table
- How grouped-query attention (GQA) lets several query heads share one KV head
- How split-K (flash-decoding) divides one sequence across programs and combines partial softmaxes exactly
- Why empty splits need a finite "minus infinity" and a masked probability, not `-inf`

## What you need to know first

- Chapter 14: the online-softmax rescaling identity
- Chapter 21: forward attention, which this chapter specializes to one query token
- Chapter 3 and 6: strides, pointers and the pointer-stride contract
- Chapter 23: the bandwidth-bound reasoning style

## 24.1 [FOUNDATIONAL] The KV Cache and Why Decode Is Memory-Bound

### Intuition

To generate token `t+1`, attention needs the keys and values of all `t` earlier tokens. They never change, so serving systems keep them: the KV cache. Each decode step reads the entire cache for every layer but does little arithmetic with it. Per token the cache costs:

```
bytes/token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_element
```

### Background

For a Llama-3-70B-shaped model (80 layers, 8 KV heads, head dimension 128, BF16):

| Quantity | Value |
|---|---|
| Bytes per token | 2 x 80 x 8 x 128 x 2 = 327,680 (320 KiB) |
| One 128K-token sequence | about 40 GiB |

```
bytes/token = 327680 (320 KiB); 128K tokens = 40.0 GiB
```

Check these against your own model's config before quoting them; they depend on layers, KV heads and precision.

Because every step re-reads a cache this large, decode throughput is set by HBM bandwidth. Batching helps because the weights are read once for many sequences, but the KV cache is per sequence: it does not amortize.

## 24.2 [FOUNDATIONAL] Paging: the Block Table

### Intuition

Allocating a contiguous worst-case buffer per sequence wastes memory, because most sequences end early. A paged cache hands out fixed-size pages on demand, and a **block table** maps each sequence's logical page number to a physical page.

### Background

```
 sequence 1, 9 tokens, page size 4:   logical page:   0    1    2
                                       block table:   [17] [ 3] [22]
                                                       |    |    |
 physical cache  [ page 0 | ... | page 3 | ... | page 17 | ... | page 22 | ... ]
                                     ^              ^             ^
                                     tokens 4-7     tokens 0-3    tokens 8-11 (only 8 used)
```

| Tensor | Shape | Meaning |
|---|---|---|
| `k_cache`, `v_cache` | `[num_pages, page_size, kv_heads, head_dim]` | Physical storage |
| `block_table` | `[batch, max_pages]` | Logical to physical page, per sequence |
| `seq_lens` | `[batch]` | Number of valid tokens per sequence |
| `q` | `[batch, heads, head_dim]` | One query token per sequence |

!!! warning "COMMON TRAP"
    Pages for one sequence are **not contiguous and not ordered** in physical memory. A kernel that assumes `page_id = base + i` passes tests where you allocated pages sequentially and fails in a real server. The reference test in this chapter shuffles page assignments on purpose.

## 24.3 [FOUNDATIONAL] Split-K: Filling the GPU

### Intuition

A decode step has one query per head. With a batch of 8 and 32 heads that is 256 programs if each handles one `(sequence, head)` pair, fewer than the 132 SMs times several resident programs can use well, and each program walks a very long cache serially. Split-K divides the sequence into `NSPLIT` chunks, each handled by its own program, and combines the partial results afterwards.

### Background

Each split computes, for its slice of tokens, three things: its running maximum `m`, its sum of exponentials `l`, and its unnormalized weighted value sum `o`. The exact merge is:

```
m_g   = max over splits of m_s
w_s   = exp(m_s - m_g)
out   = ( sum_s w_s * o_s ) / ( sum_s w_s * l_s )
```

This is the online-softmax identity from Chapter 14 applied across programs instead of across loop iterations, so the result is exact, not approximate.

!!! warning "COMMON TRAP"
    Do not use `-inf` for masked scores. An empty split has `m = -inf`, and `exp(-inf - (-inf))` is NaN. Use a large finite negative number (`-1e30`) and multiply the probabilities by the valid mask, so masked lanes contribute exactly zero. Then `exp(m_old - m_new)` stays defined for splits that see no tokens.

### Worked Example 24.3.1: stage 1 (partial results per split)

```python
@triton.jit
def decode_stage1(q_ptr, k_ptr, v_ptr, bt_ptr, lens_ptr,
                  part_m_ptr, part_l_ptr, part_o_ptr,
                  sm_scale, max_pages, pages_per_split,
                  stride_qb, stride_qh,
                  stride_kp, stride_kt, stride_kh,
                  stride_btb, group,
                  PAGE: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr):
    b = tl.program_id(0); h = tl.program_id(1); s = tl.program_id(2)
    kvh = h // group                                   # GQA: several q heads share a kv head
    seq_len = tl.load(lens_ptr + b)
    d = tl.arange(0, D); t = tl.arange(0, PAGE)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d).to(tl.float32)
    m = tl.full((), NEG, tl.float32); l = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    first = s * pages_per_split
    for i in range(0, pages_per_split):
        pg = first + i
        tok = pg * PAGE + t
        valid = (tok < seq_len) & (tok < max_pages * PAGE)
        page_id = tl.load(bt_ptr + b * stride_btb + tl.minimum(pg, max_pages - 1))
        kv_off = page_id * stride_kp + kvh * stride_kh + t[:, None] * stride_kt + d[None, :]
        k = tl.load(k_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + kv_off, mask=valid[:, None], other=0.0).to(tl.float32)
        sc = tl.where(valid, tl.sum(k * q[None, :], axis=1) * sm_scale, NEG)
        m_new = tl.maximum(m, tl.max(sc, axis=0))
        p = tl.where(valid, tl.exp(sc - m_new), 0.0)
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new
    idx = (b * tl.num_programs(1) + h) * NSPLIT + s
    tl.store(part_m_ptr + idx, m); tl.store(part_l_ptr + idx, l)
    tl.store(part_o_ptr + idx * D + d, acc)
```

Stage 2 loads the `NSPLIT` partials for each `(sequence, head)`, applies the merge above, and stores the output. The complete listing is in the downloadable file.

### File: `ch24_paged_decode_attention.py`

[Download the complete reference](../examples/references/part8/ch24_paged_decode_attention.py) (interpreter mode; GPU performance not established).

```bash
TRITON_INTERPRET=1 python3 ch24_paged_decode_attention.py
```

```
nsplit=1: matches reference? True  max|err|=2.38e-07
nsplit=2: matches reference? True  max|err|=2.38e-07
nsplit=4: matches reference? True  max|err|=2.38e-07
nsplit=8: matches reference? True  max|err|=2.38e-07
edge lengths (1, 8, 16) ok? True
KV bytes/token = 327680 (320 KiB); 128K tokens = 40.0 GiB
```

The test uses three sequences of lengths 1, 9 and 30 so the pages are partly filled and some splits are empty, four query heads sharing two KV heads, and shuffled physical pages.

## 24.4 Choosing the Split Count (a procedure, untested on GPU)

!!! warning "UNTESTED ON H100"
    No timing below was measured.

More splits mean more parallelism but more partial results to write and combine. To choose, sweep `NSPLIT` in {1, 2, 4, 8, 16, 32} at several context lengths (1K, 8K, 32K, 128K) and batch sizes (1, 8, 32), time with `triton.testing.do_bench`, and report achieved bandwidth against the KV bytes read. Expect short contexts to prefer few splits and long contexts at small batch to prefer many. A kernel that cannot reach a large fraction of HBM bandwidth at 128K context is the first place to look in Nsight Compute (Chapter 12).

## Chapter Summary

Decode attention reads the whole KV cache to produce one token, so it is bound by HBM bandwidth, and the cache costs about 320 KiB per token for a Llama-3-70B-shaped model, roughly 40 GiB for a 128K-token sequence. A paged cache stores fixed-size pages and a per-sequence block table, and pages must never be assumed contiguous. Split-K gives long sequences enough parallelism by computing a partial maximum, sum and weighted value per split and merging them exactly. Using a finite negative value with a masked probability keeps empty splits from producing NaN. The kernel matched a reference for several split counts, ragged lengths, grouped-query heads and shuffled pages in the interpreter; timing is left for you to measure.

## Self-Check Questions

1. Why does batching amortize the weight reads in decode but not the KV-cache reads?
2. What goes wrong if a kernel assumes a sequence's pages are consecutive in physical memory?
3. Write the merge formula for combining two splits' `(m, l, o)` results.
4. Why is `-inf` unsafe as the masked score for an empty split, and what replaces it?
5. For a model with 32 layers, 8 KV heads, head dimension 128 and FP8 (1 byte) cache entries, how many bytes does one token cost?

## Where We Go Next

Chapter 25 returns to the weights: storing them in 4 bits to cut the other half of the decode memory traffic, and measuring honestly what that costs in accuracy.

## Worked Solutions

1. Every sequence in the batch uses the same weights, so one weight read serves all of them. Each sequence has its own KV cache, so reading it is per-sequence work that grows with the batch.
2. Real allocators hand out whichever pages are free. A kernel assuming `page_id = base + i` reads the wrong memory and returns plausible but wrong attention output, with no error.
3. With `m_g = max(m_1, m_2)`, `out = (exp(m_1 - m_g) * o_1 + exp(m_2 - m_g) * o_2) / (exp(m_1 - m_g) * l_1 + exp(m_2 - m_g) * l_2)`.
4. An empty split has `m = -inf`, so `exp(m_old - m_new)` becomes `exp(-inf - (-inf))`, which is NaN. A finite value such as `-1e30` keeps the exponent defined, and multiplying probabilities by the valid mask makes masked lanes contribute exactly zero.
5. `2 x 32 x 8 x 128 x 1 = 65,536` bytes (64 KiB).
