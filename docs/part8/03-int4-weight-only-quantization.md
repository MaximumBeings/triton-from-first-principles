# Chapter 25: Weight-Only INT4 and Simulated FP4

> Decode reads every weight once per token, so shrinking the weights shrinks the dominant cost. Four bits per weight cuts that traffic to about a quarter of FP16. What it costs is accuracy, and the only honest way to report it is to measure it.

!!! info "What was and was not validated"
    The quantizer, the packing, and the Triton dequant-GEMM kernel were run in the **interpreter** (`TRITON_INTERPRET=1`, CPU) on **synthetic weights**. They verify that the kernel computes exactly the dequantized matrix product. They say **nothing** about perplexity on a real model, latency, or VRAM bandwidth. The evaluation procedure in Section 25.4 is a plan to run, not a result.

!!! warning "Hopper has no FP4 tensor cores"
    An H100 has FP8 tensor cores (Chapter 7) but no native FP4 path. On Hopper, "4-bit" means **weight-only** quantization: weights are stored in 4 bits and **dequantized to BF16 or FP16 inside the kernel**, and the math runs at 16-bit. The saving is memory capacity and bandwidth, not tensor-core throughput. Native FP4 math belongs to Blackwell; see the B200 appendix of Programming the H100. The FP4 code here is a **simulation** of the number format for studying error, not a speed path.

!!! info "Project 6 of the twelve"
    See [Appendix E](../appendix/inference-projects.md).

## What you will understand by the end of this chapter

- Why weight-only 4-bit helps decode (memory-bound) and can hurt large-batch prefill (compute-bound)
- How group-wise symmetric INT4 quantization works, and how two 4-bit values pack into one byte
- How a simple calibration step (searching the clip range per group) reduces error
- How a dequant-inside-the-GEMM kernel keeps weights in 4 bits until the last moment
- How the FP4 (E2M1) grid differs from INT4, and the NVFP4 versus MXFP4 scaling conventions
- Why perplexity alone is not enough, and what a fuller evaluation includes

## What you need to know first

- Chapter 13: the blocked matmul kernel
- Chapter 23 and 24: bandwidth-bound reasoning
- Chapter 7 of Programming the H100: FP8 on Hopper

## 25.1 [FOUNDATIONAL] Why Weight-Only 4-Bit Helps (and When It Does Not)

### Intuition

In decode the batch is small, every weight is read once per step, and arithmetic is cheap. Time is about bytes. Storing weights in 4 bits instead of 16 reads about a quarter as many bytes. In prefill the batch is large, the matrix multiply is compute-bound, and the dequantization work added to the inner loop can make the kernel slower than plain 16-bit.

### Background

| Regime | Limiting resource | Effect of weight-only INT4 |
|---|---|---|
| Decode, small batch | Memory bandwidth | Fewer bytes read: faster, in principle |
| Prefill, large batch | Tensor-core compute | Extra dequant work: can be slower |
| Model capacity | VRAM | About a quarter of the weight memory, plus scales |

```
FP16 weights:   [ 16 bits ][ 16 bits ][ 16 bits ][ 16 bits ]  ...
INT4 packed:    [ 4 ][ 4 ] -> one byte; plus one scale per group of 32 weights
```

Memory arithmetic from the test run, including 16-bit scales:

```
packed bytes: 3072 vs fp16 bytes: 12288  (+ scales 384 B at fp16) -> 0.28x of fp16
```

The scales are why the saving is 0.28x and not 0.25x.

## 25.2 [FOUNDATIONAL] Group-Wise INT4 and Calibration

### Intuition

One scale for a whole row is too coarse: a single large weight forces a large scale and crushes every small weight to zero. Giving each group of 32 consecutive weights its own scale limits the damage. Calibration then adjusts each group's scale. Round-to-nearest (RTN) sets the scale from the largest magnitude; clip search tries smaller ranges, sacrificing the occasional outlier to resolve the many small weights better.

### Background

| Step | What happens |
|---|---|
| Group | Split each row of `K` weights into groups of 32 |
| Scale | `scale = clip_ratio * max(abs(group)) / 7` |
| Round | `q = clamp(round(w / scale), -8, 7)` |
| Pack | Offset by 8 to `0..15`; low nibble holds even `k`, high nibble odd `k` |
| Calibrate | Try clip ratios 0.6 to 1.0; keep the one with the lowest squared error per group |

!!! warning "COMMON TRAP"
    Quantization error measured on random Gaussian weights is optimistic. Real weights have outliers and structure. The test below deliberately injects heavy-tailed weights so that clipping matters, but a real model still needs a real evaluation.

!!! info "Production methods"
    Round-to-nearest with clip search is the simplest calibration. Production pipelines use methods such as AWQ (activation-aware scaling) or GPTQ (second-order error compensation) with a calibration dataset. Name the method and the calibration set when you report results.

### Worked Example 25.2.1: quantize, pack, compare

```python
packed, sc = quantize_int4(w)                      # round-to-nearest
packed_c, sc_c = quantize_int4(w, clip_search=True)   # calibrated
```

```
weight rel-error  RTN=0.1349  clip-search=0.1215  clip better? True
same block size 16: INT4(clip-search)=0.0952  simulated FP4 (E2M1)=0.0897  (numerics only, not speed)
```

The second line compares INT4 and simulated FP4 at the **same** block size of 16. They are close. Do not read it as "FP4 is better": the result depends on the weight distribution, the scale format, and the block size, and it was measured on synthetic data.

## 25.3 [FOUNDATIONAL] The Dequant-Inside-the-GEMM Kernel

### Intuition

Dequantizing the whole matrix to FP16 first would allocate the large tensor you were trying to avoid. Instead, the kernel loads packed bytes, extracts the right nibble, multiplies by the group scale, and feeds the result straight into the dot product, all in registers.

### Background

```
 packed byte  = [ high nibble | low nibble ]   -> weights for k=2j+1 and k=2j
 for logical index k:   byte = packed[n, k // 2]
                        nibble = byte & 0xF   if k is even
                                 byte >> 4    if k is odd
                        w = (nibble - 8) * scale[n, k // GROUP]
```

### Worked Example 25.3.1: the kernel

```python
@triton.jit
def w4a16_kernel(a_ptr, wp_ptr, s_ptr, c_ptr, M, N, K,
                 stride_am, stride_wn, stride_sn, stride_cm,
                 GROUP: tl.constexpr, BLOCK_M: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * stride_am + ks[None, :],
                    mask=(rm[:, None] < M) & (ks[None, :] < K), other=0.0).to(tl.float32)
        byte = tl.load(wp_ptr + rn[None, :] * stride_wn + (ks[:, None] // 2),
                       mask=(rn[None, :] < N) & (ks[:, None] < K), other=0)
        nib = tl.where((ks[:, None] % 2) == 0, byte & 0xF, byte >> 4)
        sc = tl.load(s_ptr + rn[None, :] * stride_sn + (ks[:, None] // GROUP),
                     mask=(rn[None, :] < N) & (ks[:, None] < K), other=0.0)
        w = (nib.to(tl.float32) - 8.0) * sc          # dequantize in registers
        acc += tl.dot(a, w, input_precision="ieee")
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))
```

### File: `ch25_int4_weight_only.py`

[Download the complete reference](../examples/references/part8/ch25_int4_weight_only.py) (interpreter mode; GPU performance not established).

```bash
TRITON_INTERPRET=1 python3 /Users/oluwaseyiawoga/Developer/triton-from-first-principles/docs/examples/references/part8/ch25_int4_weight_only.py
```

```
kernel matches dequantized reference? True  max|err|=5.72e-06
output rel-error vs full-precision x@W.T = 0.1460
ragged shapes ok? True
```

The kernel is checked against `x @ dequant(W).T`, which isolates kernel correctness from quantization error. The 0.146 output error is the quantization error itself, on synthetic heavy-tailed weights; it is not a model-quality number. The ragged-shape check uses 3 rows of `x` and 37 output columns, so neither tile dimension divides evenly.

!!! warning "UNTESTED ON H100"
    This kernel is written for clarity, with 16-wide tiles and one nibble extracted per element. A fast version uses larger tiles, unpacks both nibbles of a byte together, and may use a different layout for the packed weights. No speed claim is made.

## 25.4 What a Fair Evaluation Includes (a plan, untested)

Perplexity on one dataset is a start and can hide failures on reasoning tasks. Report all of the following, from the same model, same data and same context length:

| Measurement | How |
|---|---|
| Perplexity | Fixed evaluation set (for example WikiText-2), BF16 baseline versus each format |
| Task accuracy | A harness such as lm-evaluation-harness on a reasoning set and a knowledge set (for example GSM8K and MMLU) |
| Memory | Weight bytes plus scale bytes, and peak VRAM at your serving batch size |
| Decode throughput | Tokens per second at batch 1, 8 and 32 |
| Prefill throughput | Tokens per second at a long prompt; expect INT4 to lose here |
| Calibration | Method (RTN, AWQ, GPTQ), dataset and number of samples |

### Number formats

| Format | Element | Block | Scale | Notes |
|---|---|---|---|---|
| INT4 (this chapter) | 4-bit integer, -8 to 7 | 32 | FP16 | Weight-only on Hopper |
| FP4 E2M1 | 1 sign, 2 exponent, 1 mantissa; magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 | 16 or 32 | varies | Native on Blackwell |
| NVFP4 | E2M1 | 16 | FP8 E4M3 per block plus a per-tensor FP32 scale | NVIDIA format |
| MXFP4 | E2M1 | 32 | E8M0 (a power of two) per block | OCP microscaling format |

Verify block sizes and scale formats against the format specifications before quoting them in your own report.

## 25.5 [FOUNDATIONAL] Self-Tests: Proving the Quantizer and the Kernel

`main()` prints a few error numbers. The `--test` flag runs a suite that fails when the quantizer or the kernel is wrong:

```bash
TRITON_INTERPRET=1 python3 ch25_int4_weight_only.py --test
```

```text
[PASS] hand case: byte 0xA3 is [-5, +2] and byte 0x08 is [0, -8] (low nibble = even k)
  [PASS] packed is uint8 [N, K/2]; scales are [N, K/group]
  [PASS] weights already on the grid (integers -7..7, amax 7 in every group) round-trip exactly
  [PASS] RTN error is at most half a scale step for every element (no clipping happens at amax/7)
  [PASS] RTN never uses code -8 (the symmetric range wastes one of the 16 codes)
  [PASS] clip-search does use -8 on this data
  [PASS] all-zero weights come back as exact zeros, no NaN
  [PASS] a constant group is reproduced (to float32 precision)
  [PASS] clip-search is no worse than RTN in EVERY group (ratio 1.0 is in its search set)
  [PASS] groups are independent: changing weights in two groups of row 3 changes no other group's result
  [PASS] RTN is scale-equivariant: quantize(2w) dequantizes to 2 x dequantize(w)
  [PASS] RTN is sign-symmetric: quantize(-w) dequantizes to -dequantize(w)
  [PASS] fp16 scales (what the 0.28x byte count assumes) change the dequantized weights by at most 2^-10 relative
  [PASS] bytes: INT4 + fp16 scales is exactly 0.28125 of fp16 for group 32 (0.25 + 1/32)
  [PASS] K not a multiple of the group, or odd K, is rejected
  [PASS] every FP4 output is scale x one of {0, .5, 1, 1.5, 2, 3, 4, 6} (up to float32 rounding)
  [PASS] FP4 quantization is idempotent
  [PASS] hand case (block max 6, scale 1): [6, .2, .3, 2.4, 2.6, 5.1, -4.9, -.2] -> [6, 0, .5, 2, 3, 6, -4, 0]
  [PASS] FP4 of zeros is zeros
  [PASS] kernel equals x @ dequant(W).T (atol 1e-4)
  [PASS] ragged M and N, K = 32..128 including K = 48 and 80 (not a multiple of the 32-wide K tile), groups 16/32/64 all match
  [PASS] hand case, nibble order: only k=0 non-zero, so the output is x[:, 0] (a swapped nibble reads k=1)
  [PASS] hand case, nibble order: only k=1 non-zero, so the output is x[:, 1]
  [PASS] hand case, scale per group: all codes +1, scales [1, 10] -> sum = 32 x 1 + 32 x 10 = 352
  [PASS] K = 48 with a 32-wide K tile: NaN scales stored just past the real K are never used (the K mask on the scale load)
  [PASS] K = 48: NaN activations stored just past the real K are never used (the K mask on the activation load)
  [PASS] bf16 activations are accepted and match the float reference to 5e-2
  [PASS] kernel is deterministic and does not modify its inputs
28/28 self-tests pass
```

Four of these are hand computations that do not use the reference at all, because a reference written by the same author can share the same mistake. Byte `0xA3` must unpack to `[-5, +2]`; a weight matrix with a single non-zero in column 0 (a low nibble) must produce `x[:, 0]`, and one in column 1 (a high nibble) must produce `x[:, 1]`; and all-ones codes with scales `[1, 10]` must sum to `32 x 1 + 32 x 10 = 352`.

**Do the tests catch bugs?** Each row is one change to a copy of the file. The number is how many of the 28 tests still passed.

| Broken copy | Tests passing | Verdict |
|---|---|---|
| kernel reads the high nibble for even `k` | 23/28 | caught |
| nibble parity flipped | 23/28 | caught |
| kernel offset 7 instead of 8 | 22/28 | caught |
| scale indexed by `k // 2` instead of `k // GROUP` | 22/28 | caught |
| weight byte indexed by `k`, not `k // 2` | 23/28 | caught |
| `acc` overwritten instead of accumulated | 22/28 | caught |
| scale load: K mask removed | 27/28 | caught by test 25 only |
| activation load: mask removed | 27/28 | caught by test 26 only |
| output store: N mask removed | heap corruption | caught |
| scale `amax / 8` instead of `amax / 7` | 21/28 | caught |
| pack shift 3 instead of 4 | 19/28 | caught |
| code offset +7 instead of +8 | 19/28 | caught |
| dequantizer high-nibble offset 7 | 18/28 | caught |
| clip search stops at 0.95 | 27/28 | caught (only by the "never worse than RTN" test) |
| clip search keeps the worse candidate | 26/28 | caught |
| FP4 block scale `/ 4` instead of `/ 6` | 26/28 | caught |
| FP4 `argmin` without `abs` | 27/28 | caught |
| FP4 sign dropped | 27/28 | caught |
| FP4 grid has 5 where it should have 4 | 27/28 | caught |
| scale load: N mask removed | 28/28 | **survives** |
| weight-byte load: mask removed | 28/28 | **survives** |
| `clamp_min(1e-8)` on the group maximum removed | 28/28 | **survives** |

Two of the "caught by one test only" rows needed a special test, and the story is worth knowing. The first draft of the suite let both mask mutants survive, because a masked-out lane's garbage is multiplied by zero and contributes nothing, so any finite garbage is harmless. The fix is to make the garbage **NaN**: tests 25 and 26 store NaN in the padding just past the real `K` (using a view whose row stride is wider than `K`), so an unmasked read turns the result NaN. That is a general trick for testing masks in the interpreter.

The three survivors are not test gaps:

- *Scale-load N mask.* Rows past `N` only feed output columns that are never stored. The mask protects an address, and the interpreter does not fault on a stray read.
- *Weight-byte mask.* A byte cannot hold NaN, and the activation for those lanes is zero, so the product is exactly zero whatever byte is read.
- *`clamp_min`.* For an all-zero group, `0 / 0` gives NaN, and converting NaN to `int8` is undefined in C. On this machine it happens to give 0, so the result is correct by luck. The clamp is there to avoid undefined behaviour, which a test on one machine cannot prove.

**A finding from the tests.** The code stores scales as `float32`, but the chapter's memory ratio (0.28x) counts them as 16-bit. Test 13 checks that rounding the scales to `float16` changes the dequantized weights by at most 2^-10 relative, and test 14 checks that the ratio is exactly 0.28125, so the claim in the text is consistent with the code and the cost of the 16-bit scale is small.

## 25.6 [FOUNDATIONAL] Experiments

Six experiments, run with `--extras`. They measure numerical error and byte counts only.

```bash
TRITON_INTERPRET=1 python3 ch25_int4_weight_only.py --extras
```

```text
Experiment A: group size vs weight error and bytes (2% outliers, N=64, K=512, fp16 scales assumed in the byte count)
group   RTN err   clip err   bytes vs fp16
   16   0.1124    0.1019     0.3125
   32   0.1441    0.1302     0.2812
   64   0.1854    0.1646     0.2656
  128   0.2312    0.1977     0.2578
  512   0.3156    0.2398     0.2520

Experiment B: outlier fraction vs weight error, group 32
outliers   RTN err   clip err   clip gain
   0.0 %   0.0977    0.0866     11.4 %
   0.5 %   0.1145    0.1024     10.5 %
   2.0 %   0.1418    0.1294      8.8 %
   5.0 %   0.1566    0.1413      9.8 %

Experiment C: error of the layer OUTPUT x @ W.T against weight error (group 32, clip search, 256 random activations)
weight rel-error 0.1302   output rel-error 0.1296   (random Gaussian x only: this does not test real activations, whose outlier channels can behave differently)

Experiment D: scale precision, group 32 RTN
fp32: weight rel-error 0.14413
fp16: weight rel-error 0.14413
bf16: weight rel-error 0.14414

Experiment E: code usage (share of weights at each end of the range), group 32
RTN          code -8: 0.000 %   code +7: 1.981 %   code -7: 1.996 %
clip-search  code -8: 1.694 %   code +7: 2.689 %   code -7: 1.184 %

Experiment F: decode bandwidth floor for the WEIGHTS of a 70e9-parameter model (assumed count; ignores embeddings, KV cache, activations)
fp16                            140.0 GB   floor  41.8 ms/token at 3.35 TB/s (datasheet)
INT4 group 32 + fp16 scales      39.4 GB   floor  11.8 ms/token at 3.35 TB/s (datasheet)
INT4 group 128 + fp16 scales     36.1 GB   floor  10.8 ms/token at 3.35 TB/s (datasheet)
batch-1 decode reads every weight once per token; at large batch or in prefill the GEMM is compute-bound and INT4 helps little
```

**A. Group size.** Error rises steadily with group size (RTN 0.112 at 16, 0.144 at 32, 0.231 at 128, 0.316 for one scale per row) while the byte ratio falls only from 0.3125 to 0.252. Going from group 32 to 128 saves 0.023 of the fp16 size and costs about 60% more weight error on this data. Group 32 is a trade, not a law.

**B. Outliers.** Clip search reduces the error by roughly 9 to 11% at every outlier rate tried, including a pure Gaussian. Its benefit does not grow with outliers here, which is a reminder that the chapter's heavy-tailed data is one synthetic choice.

**C. Output error.** With random Gaussian activations the output error (0.1296) is close to the weight error (0.1302). That is not evidence about real models, whose activations have outlier channels.

**D. Scale precision.** fp32, fp16 and bf16 scales give the same weight error to four digits (0.1441). Scale precision is not where the error comes from.

**E. Code usage.** RTN never uses code -8 and uses the other 15 codes; clip search uses -8 for 1.7% of weights because shrinking the scale pushes some negative values past -7.5.

**F. Bandwidth floor.** For an assumed 70-billion-parameter model, weights alone are 140 GB in fp16 and about 39 GB as INT4 with group-32 fp16 scales: floors of about 42 ms and 12 ms per token at the 3.35 TB/s datasheet figure. These are ceilings for batch-1 decode, ignoring the KV cache, the embedding table and activations. Measured speedups will be lower.

## 25.7 [FOUNDATIONAL] Failure Modes and Edge Cases

| Situation | What happens | Test |
|---|---|---|
| `K` not a multiple of the group size, or odd `K` | `AssertionError` in `quantize_int4` | 15 |
| `K` not a multiple of the 32-wide kernel tile (48, 80) | Correct: the tail is masked | 21, 25, 26 |
| Ragged `M` and `N` (1, 17, 37) | Correct | 21 |
| All-zero weights, constant group | Exact zeros, constant reproduced | 7, 8 |
| Round-to-nearest uses only 15 of the 16 codes | By design (symmetric range) | 5 |
| A single large weight in a group | Coarsens the whole group; clip search helps | 9, experiment B |
| bf16 activations | Accepted; output is float32 | 27 |
| Kernel changes its inputs, or is non-deterministic | Neither | 28 |
| NaN or infinity in the weights | A NaN group maximum makes the whole group NaN; not tested | not modelled |
| Weights whose group is larger than the tile (group 64 with a 32 tile) | Correct | 21 |

**Not modelled.** NaN and infinity in the weights; asymmetric quantization with a zero point; AWQ and GPTQ calibration; activation quantization; the effect on perplexity or task accuracy, which needs a real model; the packed layouts of production kernels, which reorder weights so loads are coalesced and dequantization is cheap; tile-size tuning (the kernel uses a fixed 16 x 16 x 32 tile that is far from optimal on a GPU); a transposed weight layout; native FP4 kernels, which Hopper cannot run; and all timing.

## 25.8 [FOUNDATIONAL] On Real Hardware (UNTESTED)

Everything above ran on CPU. These commands were written without access to a GPU and have **not been run by the author**; verify them against your installed versions.

```bash
# 1. Correctness on the GPU (no TRITON_INTERPRET)
python3 ch25_int4_weight_only.py --test

# 2. Compare against an fp16 matmul at several batch sizes
python3 ch25_int4_weight_only.py --bench
```

**Known gaps before `--bench` means anything:** `w4a16_matmul` allocates its output on the CPU, so it needs `device=a.device`; and the fixed 16 x 16 x 32 tile will be far slower than a tuned kernel, so a loss to fp16 `torch` at small batch tells you about the tile, not about INT4. A reasonable hypothesis to test, not a result: at batch 1 a well-tuned INT4 kernel should approach the ratio of bytes read, and at batch 128 and beyond it should lose to fp16 because it does extra work in the inner loop. For the numbers that matter, compare against a production INT4 kernel, not against this teaching kernel.

## Chapter Summary

On Hopper, 4-bit weights save memory and decode bandwidth but run through 16-bit math, because there are no FP4 tensor cores; the same dequantization overhead can slow compute-bound prefill. Group-wise INT4 gives every 32 weights their own scale and packs two values per byte, and a per-group clip search lowered error from 0.1349 to 0.1215 on heavy-tailed synthetic weights. A Triton kernel that dequantizes inside the GEMM matched the dequantized reference to about 6e-6, including ragged shapes. A simulated FP4 grid was close to INT4 at equal block size on this synthetic data, which says nothing general. Perplexity, task accuracy, memory, throughput and the calibration recipe must all be measured on a real model to claim anything.

## Self-Check Questions

1. Why can weight-only INT4 speed up decode and slow down large-batch prefill?
2. Why is the memory ratio 0.28x rather than 0.25x?
3. Why does clip search lower the error, given that it clips the largest weights?
4. The kernel is compared against `x @ dequant(W).T` and not against `x @ W.T`. What does each comparison measure?
5. Why is "perplexity went up by only 0.1" an incomplete claim?

## Where We Go Next

Chapter 26 turns from the weights to the output: forcing a model's sampled tokens to obey a grammar so that the result is always valid JSON.

## Worked Solutions

1. Decode is limited by bytes read, so fewer weight bytes help. Prefill is limited by tensor-core compute, and the extra unpack-and-scale work in the inner loop adds cost without removing a bottleneck.
2. Each group of 32 weights carries a 16-bit scale in addition to its 32 four-bit values, which adds 0.5 bit per weight on top of 4.
3. One large weight forces a large scale for the whole group, rounding the many small weights coarsely. Sacrificing the rare outlier lets the scale shrink and resolves the majority better, and the search keeps the choice only when the group's squared error falls.
4. Comparing against `x @ dequant(W).T` checks that the kernel computes the intended arithmetic. Comparing against `x @ W.T` measures the error that quantization itself introduces.
5. Perplexity can hide failures on reasoning and knowledge tasks, and says nothing about throughput, memory, which calibration method and data were used, or whether the change holds across context lengths.

## Additional Exercises

6. Round-to-nearest never uses code -8. Propose a change that uses all 16 codes and say what it costs.
7. The weight-byte mask survives mutation. Explain why no CPU test can kill it, and what tool on a GPU could.
8. Experiment E says clip search uses code -8 for 1.7% of weights. At what clip ratio does a weight at the group's most negative extreme first round to -8?
9. Compute the byte ratio against fp16 for group 128 with fp16 scales, and for group 16 with 8-bit scales.
10. Using experiment A, argue for or against group 16 over group 32 for a model whose memory budget is the binding constraint.

## Additional Solutions

6. Choose the scale as `amax / 8` when the largest-magnitude element of the group is negative and `amax / 7` otherwise, so a negative extreme lands exactly on -8. This gains resolution in about half the groups, and costs a branch in the quantizer and a changed test 5, since -8 becomes legal for RTN.
7. The product is exactly zero for those lanes whatever byte is read, and a byte cannot hold NaN, so the output cannot differ. Only the read itself differs, which is visible to a memory checker such as `compute-sanitizer --tool memcheck` on a GPU and to nothing in the interpreter.
8. With ratio `r`, the extreme maps to `-7/r`. It rounds to -8 when `7/r >= 7.5`, so `r <= 0.9333`. Below that ratio it keeps going negative and is clamped at -8.
9. Group 128 with fp16 scales: `0.25 + 1/128 = 0.2578`. Group 16 with 8-bit scales: each weight carries `1/16` byte of scale, which is `0.03125` of the 2-byte fp16 weight, so `0.25 + 0.03125 = 0.28125`, the same as group 32 with fp16 scales.
10. Group 16 costs 0.3125 of fp16 against 0.2812 for group 32, about 11% more memory, and cuts RTN weight error from 0.144 to 0.112 on this data. If memory is binding, that 11% is exactly what you are short of, so the answer is to prefer group 32 and take clip search, which gives a smaller reduction (0.144 to 0.130) for no extra bytes. If the error budget is binding rather than memory, group 16 wins. The data here is synthetic, so confirm on your model.

## Files and Sources

- `examples/references/part8/ch25_int4_weight_only.py`: quantizer, FP4 simulation, kernel, `main()`, `--test` (28 self-tests), `--extras` (experiments A to F), `--bench` (GPU only, untested, with the gaps noted above).
- Outputs shown were produced with `TRITON_INTERPRET=1` on CPU, Triton 3.9.0 and torch 2.15.
- The 3.35 TB/s figure is the H100 SXM datasheet number, used only as a ceiling. The 70-billion-parameter count in experiment F is an assumed round number.
