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

## 24.5 [FOUNDATIONAL] Self-Tests: Proving the Kernel Is Right

`main()` shows the kernel matching a reference on a few shapes. The `--test` flag runs a suite built to fail when the kernel is wrong:

```bash
TRITON_INTERPRET=1 python3 ch24_paged_decode_attention.py --test
```

```text
[PASS] 14 lengths (1..33, on and off page boundaries) x nsplit 1/2/4/8 match the fp64 reference (atol 1e-5)
  [PASS] the split count does not change the answer: nsplit 1 vs 8 agree to 1e-6
  [PASS] relabelling every physical page (cache permuted, block table remapped) leaves the output bit-identical
  [PASS] NaN in every slot beyond a sequence's length (partial last pages and unowned pages) does not leak
  [PASS] unused block-table entries holding an out-of-range page id (10^6) are never dereferenced
  [PASS] MHA (group 1), GQA (group 4) and MQA (1 kv head), D = 16/32/64, page = 4/8/16 all match
  [PASS] one token: softmax of one score is 1, so the output is exactly that token's V
  [PASS] identical keys: every weight is 1/11, so the output is the mean of the 11 V rows (hand computation, not the reference)
  [PASS] one key with a score 100 above the rest takes all the weight: output equals that token's V
  [PASS] scores in the hundreds (q x 200) do not overflow exp: finite and equal to the reference
  [PASS] every output coordinate lies between the min and max of that sequence's V (a softmax-weighted average)
  [PASS] reversing the page order of full sequences leaves the output unchanged: position lives in K (RoPE), not in the cache layout
  [PASS] rewriting sequence 0's pages leaves sequences 1 and 2 bit-identical (and changes sequence 0)
  [PASS] bf16 inputs: output is bf16 and within 2e-2 of the fp32 reference
  [PASS] two runs are bit-identical
  [PASS] q, K, V, block table and lengths are not modified
  [PASS] seq_len 40 with a 3-page table attends to the 12 tokens the table covers (same as seq_len 12): silently clamped, see text
  [PASS] seq_len 0 returns NaN (0/0 in stage 2): documented limitation, the scheduler must not launch empty sequences
  [PASS] nsplit not a power of two is rejected
  [PASS] query heads not divisible by kv heads is rejected
20/20 self-tests pass
```

The interpreter prints a `RuntimeWarning: invalid value encountered in divide` while this runs. It comes from test 18, which divides zero by zero on purpose (see 24.7).

The suite checks several properties that do not need the reference at all, because a reference written by the same author can share the same misunderstanding: with identical keys the output is the mean of V (a hand computation); with one token the output is that token's V; one dominant key takes all the weight; every output coordinate lies inside the range of V; relabelling the physical pages leaves the result bit-identical; and rewriting one sequence's pages cannot change another's output.

**Do the tests catch bugs?** Each row is one change to a copy of the kernel. The number is how many of the 20 tests still passed.

| Broken copy | Tests passing | Verdict |
|---|---|---|
| tokens beyond `seq_len` not masked | segfault | caught (reads the poisoned page id) |
| V load unmasked | segfault | caught |
| split offset ignored (`tok = i * PAGE + t`) | segfault | caught |
| block table ignored (page index used as page id) | 9/20 | caught |
| `NEG = -inf` instead of `-1e30` | 5/20 | caught: empty splits give `exp(-inf - -inf) = NaN` |
| `kvh = h` (no GQA mapping) | 11/20 | caught |
| `sm_scale` dropped | 16/20 | caught |
| `l` not rescaled by `alpha` | 15/20 | caught |
| `acc` not rescaled by `alpha` | 14/20 | caught |
| stage 2 weights `exp(m)` instead of `exp(m - m_g)` | 18/20 | caught (overflow at large scores) |
| stage 2 denominator ignores the weights | 14/20 | caught |
| splits per program rounded down | 14/20 | caught |
| capacity clamp `tok < max_pages * PAGE` removed | 19/20 | caught by test 17 only |
| `p` not masked with `valid` | 19/20 | caught by test 18 only: see below |
| scores not masked to `NEG` | 19/20 | caught by test 10 only |
| `tl.minimum(pg, max_pages - 1)` clamp removed | 20/20 | **survives**: see below |

Three of these deserve a sentence.

*`p` not masked.* The explicit `tl.where(valid, ..., 0.0)` is redundant for every non-empty sequence, because masked scores are already `-1e30` and `exp` of that is 0. It only matters for an empty sequence. Keep it for clarity, but know that only the empty-sequence test notices it.

*Scores not masked.* Only the large-score test notices. Masked lanes then score 0, which can pull the running maximum up and lets `exp` of the real scores underflow. The mask protects the numerical stabiliser, not the sum.

*The block-table clamp survives.* `tl.minimum(pg, max_pages - 1)` stops the kernel from reading block-table entries past the end of a row when `nsplit` does not divide the page count. The value read is masked out either way, so the output is the same. What the clamp prevents is an out-of-bounds read, and the interpreter does not fault on that: it quietly reads the neighbouring row. On a GPU the unclamped read could touch unmapped memory at the end of the buffer. This test suite cannot see that, so it is listed under "not modelled" below and is a reason to run the tests under `compute-sanitizer` on a real GPU.

## 24.6 [FOUNDATIONAL] Experiments

Four experiments, run with `--extras`. None measures speed.

```bash
TRITON_INTERPRET=1 python3 ch24_paged_decode_attention.py --extras
```

```text
Experiment A: programs launched by stage 1 = batch x q_heads x nsplit (64 q heads; an H100 SXM has 132 SMs, datasheet)
batch  nsplit=1   2    4    8   16   32
    1     64    128    256    512   1024   2048
    4    256    512   1024   2048   4096   8192
   16   1024   2048   4096   8192  16384  32768
   64   4096   8192  16384  32768  65536  131072
any entry below ~132 leaves SMs idle; a single sequence (batch 1) with 64 heads already gives 64 programs, so it needs nsplit >= 4 to pass 132

Experiment B: error against fp64 vs context length (B=1, H=2, KVH=1, D=16, PAGE=16, fp32 kernel)
tokens  nsplit=1     nsplit=8
    64   1.34e-07   1.34e-07
   256   4.47e-08   4.47e-08
  1024   8.94e-08   8.94e-08

Experiment C: wasted page iterations in a mixed batch (the grid is sized by the LONGEST sequence), PAGE=16, nsplit=4
batch                 live pages  iterations  fully-masked share
uniform 2000 x 8            1000        1024              2.3 %
long tail                    422        2016             79.1 %
one long + 7 short           528        4000             86.8 %
masked iterations load nothing (the loads are masked) but still cost loop and issue time; per-sequence split counts or sorting by length reduce it

Experiment D: KV bytes per token and per decode step (bf16; model shapes are ASSUMED round numbers, check your model's config)
model shape                          B/token   per-step read at 8K   at 128K   floor at 128K (3.35 TB/s)
70B-like: 80 layers, 8 kv heads, D=128   320 KiB         2.50 GiB     40.0 GiB      12.8 ms
8B-like: 32 layers, 8 kv heads, D=128    128 KiB         1.00 GiB     16.0 GiB       5.1 ms
no GQA: 32 layers, 32 kv heads, D=128    512 KiB         4.00 GiB     64.0 GiB      20.5 ms
every decode step of ONE sequence reads its whole cache once, so the floor per generated token is bytes / bandwidth
```

**A. Programs launched.** Stage 1 launches `batch x q_heads x nsplit` programs. With 64 query heads, a single sequence at `nsplit = 1` gives 64 programs, fewer than the 132 SMs of an H100 SXM (datasheet figure), so most of the chip idles. `nsplit = 4` is the first value that passes 132. At batch 16 and above the grid is already large and extra splits only add partial-result traffic.

**B. Accuracy against context length.** The `float32` kernel's worst error against `float64` is about 1e-7 at 64, 256 and 1024 tokens, and splitting 1 way or 8 ways gives the same figure to the digits shown. That is the `float32` rounding floor, so at these lengths the split-combine step costs no accuracy. The experiment does not show behaviour at 128K tokens, which is too slow for the interpreter.

**C. Wasted iterations.** The grid is sized by the longest sequence in the batch. A uniform batch wastes 2.3% of page iterations; a long-tail batch wastes 79%; one 8000-token sequence next to seven short ones wastes 87%. Masked iterations load no bytes, but they still cost loop overhead and issue slots, so mixed-length batches are where per-sequence split counts or length-sorted batching would pay off. Whether they do is a question for a profile.

**D. KV bytes per token.** With the chapter's Llama-3-70B-shaped model the cache is 320 KiB per token, 40 GiB at 128K tokens, and a bandwidth floor of about 12.8 ms per generated token at 3.35 TB/s for that one sequence. The 8B-like and no-GQA rows use round-number shapes chosen for contrast, not a specific model's configuration; check your own model's config. The no-GQA row shows what grouped-query attention buys: a 4x smaller cache for the same number of query heads.

## 24.7 [FOUNDATIONAL] Failure Modes and Edge Cases

| Situation | What happens | Test |
|---|---|---|
| `seq_len = 1` | Output is exactly that token's V | 7 |
| Length ends mid-page, or exactly on a page boundary | Correct | 1 |
| `nsplit` larger than the number of pages | Extra splits are empty and contribute weight 0 | 1, 2 |
| Garbage (NaN) in unused slots of a page | Not read | 4 |
| Out-of-range ids in unused block-table entries | Not dereferenced | 5 |
| Physical page order arbitrary | No effect | 3, 12 |
| Very large attention scores | Finite and correct | 10 |
| `seq_len` larger than the block table covers | Silently clamped to the table's capacity | 17 |
| `seq_len = 0` | **Returns NaN** (0/0 in stage 2) | 18 |
| `nsplit` not a power of two; query heads not divisible by kv heads | `AssertionError` | 19, 20 |
| `seq_len` negative, or block-table entries negative inside the live range | Not checked | not modelled |

Two of these are decisions you must make. A scheduler must never launch an empty sequence, or must replace NaN with zeros afterwards; exercise 8 asks you to make the kernel handle it. And a length beyond the table is clamped without any error, which hides a scheduler bug: a request that has outgrown its allocated pages will quietly attend to a truncated context. A production kernel should assert on the host that `seq_len <= max_pages * PAGE`.

**Not modelled.** Out-of-bounds block-table reads (see 24.5), `D` or `PAGE` that is not a power of two (`tl.arange` rejects it), `float8` caches and dequantisation, per-sequence variable split counts, prefix-sharing between sequences (two block tables pointing at the same physical page, which should work but is not tested), sliding-window or sparse attention, CUDA-graph capture, the wrapper's scratch tensors being created on the CPU, and all timing.

## 24.8 [FOUNDATIONAL] On Real Hardware (UNTESTED)

Everything above ran on CPU. These commands were written without access to a GPU and have **not been run by the author**; verify the flags against your installed versions.

```bash
# 1. Correctness on the GPU (no TRITON_INTERPRET)
python3 ch24_paged_decode_attention.py --test

# 2. Out-of-bounds and race checks, which the interpreter cannot do
compute-sanitizer --tool memcheck python3 ch24_paged_decode_attention.py --test

# 3. Split-count sweep
python3 ch24_paged_decode_attention.py --bench
```

`--bench` times the kernel for batch 4, 32 query heads, 8 kv heads, `D = 128`, page 16 and 8192 tokens, for `nsplit` from 1 to 32, and prints achieved GB/s against the bytes of K and V read once. **Known gap:** the wrapper allocates its scratch tensors with `torch.empty` and no `device`, so on a GPU it fails until you create them on `q.device`; fix that line first. A reasonable hypothesis to test is that achieved bandwidth rises with `nsplit` while there are fewer programs than SMs and then flattens or falls as partial-result traffic grows. For a profile, `ncu --set full` on the stage 1 kernel and read the memory workload section, and compare the result with the datasheet peak.

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

## Additional Exercises

6. Explain why reversing the page order of a full sequence leaves the output unchanged (test 12), and say what would break if the kernel applied rotary embeddings itself using the cache position.
7. Experiment C shows 79% masked iterations for a long-tail batch. Sketch two changes that reduce it and say what each costs.
8. Change the kernel so that `seq_len = 0` returns zeros. Which line do you change, and which of the mutation results above does your change interact with?
9. Add a host-side assertion that catches a `seq_len` larger than the block table covers. Why is it better on the host than in the kernel?
10. Design a test that detects the missing block-table clamp. What tool or setup would you need?

## Additional Solutions

6. Attention without positions is a weighted average over a set, so visiting pages in another order only changes the order of a sum. That works because positions were already folded into K before it was cached. If the kernel rotated K by its position in the cache, a reversed table would assign different positions and change the answer, and relabelling physical pages would also have to be reconsidered.
7. Sort or bucket sequences by length so each launch has similar lengths, which costs scheduling complexity and may split a batch into several launches. Or give each sequence its own split count and have programs for an unused split exit early, which costs a more complicated grid and combine step. Both need profiling to know whether they help.
8. In stage 2, guard the division: `out = tl.sum(...) / tl.maximum(denom, 1e-30)`. The numerator is also zero for an empty sequence, so the result is zeros. Test 18 must then expect zeros instead of NaN. A consequence worth noticing: the "`p` not masked" mutant was caught only by test 18, so after this change no test catches it, and you would need a new test that distinguishes it (for example by inspecting the stage 1 partial sums).
9. `assert int(seq_lens.max()) <= block_table.shape[1] * PAGE` before launch. A host check runs once and gives a Python error with the offending value, whereas a check inside the kernel runs in every program, costs time, and can at best write a flag.
10. Allocate the block table with a guard region, run under `compute-sanitizer --tool memcheck` on a GPU, and use a case where `nsplit * pps` exceeds `max_pages`. The sanitizer reports the out-of-bounds read of the block table. The interpreter cannot, so no CPU-only test can.

## Files and Sources

- `examples/references/part8/ch24_paged_decode_attention.py`: kernels, `main()`, `--test` (20 self-tests), `--extras` (experiments A to D), `--bench` (GPU only, untested, with the scratch-device gap noted above).
- Outputs shown were produced with `TRITON_INTERPRET=1` on CPU, Triton 3.9.0 and torch 2.15.
- The 132-SM and 3.35 TB/s figures are the H100 SXM datasheet numbers, used only as ceilings.
