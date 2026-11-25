# Chapter 23: Fused Residual + RMSNorm, Profiled Against PyTorch

> Every transformer layer ends the same way: add the sublayer's output to the residual stream, then normalize. Written in eager PyTorch that is five memory round trips over a tensor that never needed to leave the chip. This chapter writes the one-kernel version, proves it correct, and shows how to measure whether it earned its keep.

!!! info "What was and was not validated"
    The kernel and every output block below were run in the Triton **interpreter** (`TRITON_INTERPRET=1`, CPU). That validates numerics and masking. It says nothing about speed. The profiling commands in Section 23.3 are provided but **have not been run on an H100**; no bandwidth or speedup figures are claimed. Replace them with your own measurements.

!!! info "Project 1 of the twelve"
    This chapter is Project 1 of the [twelve inference-engineering projects](../appendix/inference-projects.md): *write a fused kernel in Triton, profile it with Nsight Compute, benchmark it against PyTorch.*

## What you will understand by the end of this chapter

- Why RMSNorm is a **bandwidth-bound** operation, and what its speed limit is before you write a line of code
- How to fuse a residual add and an RMSNorm into one kernel that reads each input once and writes each output once
- Why the kernel must also write the **updated residual**, not just the normalized output
- How to count memory traffic for the eager and fused versions, and what ratio that predicts
- How to check the kernel against a reference with awkward row widths, and against a near-zero row where `eps` placement matters
- Which Nsight Compute numbers to read, and what each would tell you

## What you need to know first

- Chapter 12: elementwise kernels, and why fusion saves memory traffic
- Chapter 14: row-wise reductions with `tl.sum` and masking
- Chapter 20: the LayerNorm kernel and its `eps` trap, of which this is the simpler cousin
- Chapter 19: benchmarking correctly (warm-up, synchronization, medians)

## 23.1 [FOUNDATIONAL] What RMSNorm Computes and Why It Is Bandwidth-Bound

### Intuition

LayerNorm subtracts the row mean and divides by the standard deviation. RMSNorm skips the mean: it divides each element by the root-mean-square of its row, then multiplies by a learned per-column weight. Fewer reductions, and nearly every modern decoder model uses it.

For a row `h` of width `N`:

```
rms   = sqrt( mean(h * h) + eps )
y[j]  = h[j] / rms * w[j]
```

Count the arithmetic: a handful of floating-point operations per element. Count the memory: every element must be read from HBM and written back. The ratio of work to bytes is tiny, so the GPU spends its time waiting for memory, not computing. That makes the speed limit simple:

```
time  >=  bytes moved  /  HBM bandwidth
```

### Background

```
                 residual add + RMSNorm, per row of N elements

   eager PyTorch (each op is a kernel; each arrow is HBM traffic)

   x,res --> [add] --> h --> [pow] --> h^2 --> [mean] --> ms
                       |                                   |
                       +--> [mul rsqrt(ms)] --> t --> [mul w] --> y

   fused (one kernel; h lives in registers)

   x,res --> [ add -> square -> sum -> rsqrt -> scale ] --> y, new_res
```

| | Eager PyTorch | Fused (this chapter) |
|---|---|---|
| Kernel launches per layer | about 5 | 1 |
| Element transfers per element (read + write) | about 10 | 4 |
| Predicted traffic ratio | 2.5x the fused version | 1x |

The 10 versus 4 count comes from adding up each operation's reads and writes: the add reads two tensors and writes one (3), the square reads one and writes one (2), the mean reads one (1), the multiply by the inverse RMS reads one and writes one (2), the multiply by the weight reads one and writes one (2). The fused kernel reads `x` and `res` (2) and writes `y` and the new residual (2). The weight vector is small and reused across rows, so it is ignored in both counts.

A prediction of 2.5x less traffic is an upper bound on speedup for a purely bandwidth-bound op. Real results are lower because caches absorb some eager re-reads and launch overhead is small at large sizes. Treat 2.5x as the ceiling you compare against, never as a result.

### Worked Example 23.1.1: the traffic arithmetic

```python
n = 4096 * 8192                      # a 4096-token batch, hidden size 8192
fused, eager = 4 * n * 2, 10 * n * 2 # bf16 = 2 bytes per element
print(f"fused={fused/1e6:.0f} MB, eager={eager/1e6:.0f} MB, ratio={eager/fused:.1f}x")
```

```
fused=268 MB, eager=671 MB, ratio=2.5x
```

On an H100 SXM with about 3.35 TB/s of HBM3 bandwidth, 268 MB is a lower bound of roughly 80 microseconds for the fused kernel, and 671 MB is roughly 200 microseconds for the eager chain. Those are floors, not predictions: no kernel beats its bandwidth floor, and many land at 70 to 90 percent of it.

## 23.2 [FOUNDATIONAL] The Fused Kernel

### Intuition

One program handles one row. It loads `x` and `res`, adds them in `float32`, squares and sums the result, computes the inverse RMS, scales, and stores. The sum `h` is also stored as the **new residual**, because the next layer needs it. A kernel that returned only `y` would force you to recompute `h` in a second pass, destroying the saving.

### Background

| Design choice | What this chapter does | Why |
|---|---|---|
| Accumulation precision | Upcast to `float32` before the square | `bf16` squares lose precision across a wide row |
| `eps` placement | Inside the square root | Matches PyTorch convention; see the trap below |
| Row handling | One program per row, `BLOCK = next_power_of_2(N)` | Whole row in one reduction, no cross-program sync |
| Row width limit | Asserted at 65536 | One program must hold the whole row |
| Masked lanes | Load `0.0` | Zeros add nothing to the sum of squares; divide by the true `N` |

!!! warning "COMMON TRAP"
    Because masked lanes are loaded as `0.0`, `tl.sum(h * h)` is correct, but the mean must divide by `n_cols`, **not** `BLOCK`. Dividing by `BLOCK` passes every test where the width is a power of two and silently gives wrong answers for widths such as 100 or 1000.

### Worked Example 23.2.1: the kernel

```python
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
```

The Python wrapper allocates `y` and `new_res`, picks `BLOCK`, and launches one program per row.

### File: `ch23_residual_rmsnorm.py`

[Download the complete reference](../examples/references/part8/ch23_residual_rmsnorm.py) (interpreter mode; GPU performance not established).

```bash
TRITON_INTERPRET=1 python3 ch23_residual_rmsnorm.py
```

```
4x64: y ok=True  new_residual ok=True
5x100: y ok=True  new_residual ok=True
3x1000: y ok=True  new_residual ok=True
near-zero row ok: True
analytic bytes (bf16, 4096x8192): fused=268 MB, eager=671 MB, ratio=2.5x
```

The widths 100 and 1000 are deliberately not powers of two; they are the cases the divide-by-`BLOCK` bug would break.

## 23.3 [FOUNDATIONAL] Profiling and Benchmarking It Honestly

### Intuition

A correct kernel is not yet a fast kernel, and a fast kernel is not yet proven fast. The question to answer is how close the kernel gets to its bandwidth floor, and how that compares with PyTorch.

### Background

Measure achieved bandwidth as bytes moved divided by kernel time, and report it as a percentage of the device's peak. For this kernel:

```
achieved GB/s = (4 * rows * cols * bytes_per_element) / kernel_seconds / 1e9
percent of peak = achieved GB/s / peak GB/s
```

!!! warning "UNTESTED ON H100"
    Nothing in this section was run on a GPU. Use it as a procedure, not as a record.

| Step | Command or action | What to read |
|---|---|---|
| Benchmark | `triton.testing.do_bench` on the fused kernel and on `torch` eager | Median time, over several hidden sizes (1024, 4096, 8192, 16384) |
| Profile | `ncu --set full -k regex:residual_rmsnorm python3 bench.py` | Memory throughput section |
| Check the roofline | Compare DRAM throughput with peak | Is it near the bandwidth ceiling? |
| Check stalls | Warp state statistics | Long scoreboard stalls mean waiting on memory, as expected here |
| Check wide rows | Registers per thread, occupancy | Very wide rows spill registers and drop throughput (Chapter 11) |

Report a table with one row per hidden size: eager time, fused time, speedup, achieved GB/s, percent of peak. Include the predicted ceiling of 2.5x beside the measured speedup. If the measured speedup is far below it, the Nsight Compute memory section tells you whether you are bandwidth-limited, latency-limited or launch-limited.

## 23.4 [FOUNDATIONAL] Self-Tests: Proving the Kernel Is Right

The three checks in `main()` are a demonstration. The `--test` flag turns them into a suite that fails loudly:

```bash
TRITON_INTERPRET=1 python3 ch23_residual_rmsnorm.py --test
```

```text
[PASS] fp32: 30 shapes (rows 1/2/7 x widths 1..1000, incl. every power-of-two neighbour) match the fp64 reference
  [PASS] output dtypes follow the input dtype
  [PASS] bf16: new residual within 1 bf16 ulp of the fp32 sum (the interpreter truncates, a GPU rounds to nearest; see text)
  [PASS] bf16: y within 2 bf16 ulps (rel 1.6e-2) of the fp32 reference
  [PASS] fp16 inputs work and keep their dtype
  [PASS] width 1: y = h / sqrt(h^2 + eps) * w = 2 (h = 4)
  [PASS] all-zero rows give exactly 0, not NaN (eps keeps the division finite)
  [PASS] large values (1e4) do not overflow: result is ~1 everywhere
  [PASS] fp16 near its max (3e4): fp32 accumulation avoids overflow of the sum of squares
  [PASS] linear in the weight: y(2w) == 2 y(w)
  [PASS] scale invariance when eps is negligible: y(10x, 10r) ~ y(x, r)
  [PASS] rows are independent: permuting rows permutes the output
  [PASS] a NaN poisons only its own row
  [PASS] inputs are not modified
  [PASS] y and new_residual are separate buffers
  [PASS] eps matters for a near-zero row (h=1e-4): eps=1e-6 gives 1e-4/sqrt(1.01e-6)=0.0995, eps=1e-2 gives 1e-4/0.1=0.001
  [PASS] a non-contiguous input is rejected (AssertionError)
  [PASS] a width above 65536 is rejected
  [PASS] a width of 0 is rejected
  [PASS] shape mismatch between x and residual is rejected
20/20 self-tests pass
```

Every run on CPU goes through the Triton interpreter, so these tests establish numerics, masking and argument checking. They say nothing about speed.

**Do the tests actually catch bugs?** A test suite that passes on a broken kernel is worse than none. Each line below is one change to a copy of the kernel, and the number is how many of the 20 tests still passed (20 means the bug survived):

| Broken copy | Tests passing | Verdict |
|---|---|---|
| divide by `BLOCK` instead of `n_cols` | 18/20 | caught |
| `eps` added outside the square root | 14/20 | caught |
| `eps` dropped | 17/20 | caught |
| updated residual never stored | 17/20 | caught |
| masked lanes load `1.0` for `x` | 17/20 | caught |
| residual add skipped | 17/20 | caught |
| weight dropped from the output | 16/20 | caught |
| weight indexed by row offset instead of column | 15/20 | caught |
| `y` stored into the residual buffer | 7/20 | caught |
| row stride ignored | 11/20 | caught |
| store mask removed (`y` or new residual) | crash | caught: the interpreter aborts on the out-of-bounds write |
| masked lanes load `1.0` for the **weight** | 20/20 | survives: an equivalent mutant (see below) |
| no explicit `float32` cast on `x` | 20/20 | survives: an equivalent mutant (see below) |

The two survivors are not test gaps. A masked weight lane is multiplied into an output lane that is never stored and never enters the sum, so its value cannot be observed. Dropping the cast on `x` changes nothing because `x + r` is promoted to `float32` anyway when `r` is cast. Both lines are still worth keeping for clarity, but you should know they are not load-bearing.

**One finding from writing the bf16 test.** The first draft asserted that the stored residual equals `bf16(x + r)` exactly. It failed on 78 of 300 elements. The cause: the Triton interpreter converts `float32` to `bfloat16` by truncating, while a GPU converts with round-to-nearest-even. The two differ by one unit in the last place. The final test therefore asserts "within one bf16 ulp of the float32 sum", which is true on both. If you add an exact-equality bf16 test of your own, expect it to pass on a GPU and fail in the interpreter, or the reverse.

## 23.5 [FOUNDATIONAL] Experiments

Four small experiments, run with `--extras`. None measures speed.

```bash
TRITON_INTERPRET=1 python3 ch23_residual_rmsnorm.py --extras
```

```text
Experiment A: why accumulate in fp32? Serial sum of squares of a bf16 row, relative error of the mean-square vs fp64
(torch.sum on bf16 already accumulates in fp32 internally, so a Python loop is used to model a naive bf16 accumulator)
width   serial bf16 accumulator   fp32 accumulator
  256                4.13e-02           2.28e-08
 1024                1.85e-01           4.42e-08
 4096                5.06e-01           2.87e-08

Experiment B: lanes wasted by BLOCK = next_power_of_2(width), for real hidden sizes
hidden  BLOCK  wasted lanes
  2048   2048    0.0 %
  2560   4096   37.5 %
  3584   4096   12.5 %
  4096   4096    0.0 %
  5120   8192   37.5 %
  6144   8192   25.0 %
  7168   8192   12.5 %
  8192   8192    0.0 %
 14336  16384   12.5 %

Experiment C: bandwidth floor by token count (hidden 8192, bf16, H100 SXM 3.35 TB/s datasheet)
tokens  fused MB  eager MB  fused floor us  eager floor us
     1      0.07      0.16            0.02            0.05
    16      1.05      2.62            0.31            0.78
   256     16.78     41.94            5.01           12.52
  4096    268.44    671.09           80.13          200.32
(at 1 token the floor is a few hundredths of a microsecond: launch and latency, not bandwidth, decide the real time)

Experiment D: error against fp64 over 20 seeds, fp32 kernel, 8 x 1000
worst max|error| = 1.41e-06
```

**A. Why accumulate in `float32`.** `torch.sum` on a `bfloat16` tensor already accumulates in `float32` internally, so calling it would hide the effect. The experiment therefore adds the squares one at a time in `bfloat16`, as a naive kernel would. Relative error of the mean-square grows with width (4% at 256, about 19% at 1024, about 51% at 4096 in the run above) while the `float32` accumulator stays near 1e-8. The error grows because once the running sum is large, adding a small square falls below the `bfloat16` spacing and is lost.

**B. Wasted lanes.** The kernel rounds the width up to a power of two. For Llama-style hidden sizes, 2560 and 5120 waste 37.5% of the lanes. Masked lanes cost issue slots but not memory traffic, so this shows up as lower arithmetic efficiency, not extra bytes.

**C. Bandwidth floor by token count.** At 4096 tokens and hidden 8192, the floor is about 80 microseconds fused versus 200 eager (datasheet 3.35 TB/s, so a ceiling, not a measurement). At one token the floor is a few hundredths of a microsecond, which is far below the launch cost, so for decode-sized batches this kernel is launch-bound and the 2.5x traffic ratio does not translate into a 2.5x speedup.

**D. Error against `float64`.** Over 20 seeds the worst absolute error of the `float32` kernel on an 8 by 1000 problem was 1.41e-06, consistent with `float32` rounding.

## 23.6 [FOUNDATIONAL] Failure Modes and Edge Cases

| Situation | What happens | Where it is checked |
|---|---|---|
| Width not a power of two (1, 3, 100, 1000) | Correct: masked lanes are excluded from the sum and divided out via `n_cols` | test 1 |
| Width 1 | Correct: `y = h / sqrt(h^2 + eps) * w` | test 6 |
| All-zero row | Output exactly 0, no NaN, because `eps` keeps the denominator positive | test 7 |
| Large values (1e4 in `float32`, 3e4 in `float16`) | No overflow: squares are formed in `float32` | tests 8, 9 |
| NaN in one element | Poisons only that row | test 13 |
| Non-contiguous input, mismatched shapes, width 0, width above 65536 | `AssertionError` before launch | tests 17 to 20 |
| Width above 65536 | Rejected: one program holds the whole row in one block. A real model with a wider row needs a two-pass kernel | test 18 |
| Hidden size not divisible by the vector width, or rows not 16-byte aligned | Not tested here | not modelled |

**Not modelled.** Strided rows (`stride_row` other than `n_cols`) are accepted by the kernel signature but the wrapper rejects non-contiguous input, so that path is only exercised by the stride mutation above. Alignment effects on vectorised loads, `float8` inputs, in-place operation (`y` aliasing `x`), multi-GPU or tensor-parallel norms that need a cross-device reduction, and any timing are not covered.

## 23.7 [FOUNDATIONAL] On Real Hardware (UNTESTED)

Everything above ran on CPU. The commands below were written without access to a GPU and have **not been run by the author**. Treat the exact flags as things to verify against your installed versions.

```bash
# 1. Confirm the kernel is correct on the GPU (no TRITON_INTERPRET)
python3 ch23_residual_rmsnorm.py
python3 ch23_residual_rmsnorm.py --test

# 2. Benchmark fused against the eager reference at four hidden sizes
python3 ch23_residual_rmsnorm.py --bench
```

`--bench` prints eager time, fused time, speedup and achieved GB/s for hidden sizes 1024 to 16384 at 4096 tokens, using `triton.testing.do_bench`. The `reference` function upcasts to `float32` and runs several kernels, so for a fair comparison also time the RMSNorm module your model actually uses.

What to expect, as a hypothesis to test and not a result: a speedup somewhat below the 2.5x traffic ratio at large token counts, and little or no speedup at a few tokens, where launch dominates. To see why a run falls short, use Nsight Compute's memory workload section (for example `ncu --set full python3 ch23_residual_rmsnorm.py --bench`, verify the flags for your version) and look at achieved bandwidth against the datasheet peak.

Also re-run the `--test` suite on the GPU. The bf16 residual test is written to pass under both rounding behaviours, but this has only been confirmed in the interpreter.

## Chapter Summary

RMSNorm does a few operations per element, so it is limited by HBM bandwidth, and its floor is bytes moved divided by bandwidth. Fusing the residual add into the normalization cuts element transfers from about 10 to 4 per element, a predicted ceiling of 2.5x on traffic, provided the kernel also writes the updated residual. The kernel accumulates in `float32`, divides by the true row width rather than the block width, and keeps `eps` inside the square root; it passes 20 self-tests in the interpreter, and each of ten deliberately broken copies of the kernel fails at least one of them. Speed was not measured: the benchmark and Nsight Compute procedure are given for you to run on an H100.

## Self-Check Questions

1. Why is the speed limit of this kernel set by memory bandwidth and not by arithmetic?
2. Why must the fused kernel store the updated residual `h`, and what would you lose if it stored only `y`?
3. The test widths include 100 and 1000. What specific bug do they catch that width 64 would miss?
4. The analytic ratio is 2.5x. Name two reasons the measured speedup could be lower.
5. Why is accumulation done in `float32` when the tensors are `bfloat16`?

## Where We Go Next

Chapter 24 moves from normalization to attention at inference time: one new query token attending over a long, paged KV cache. It is the same bandwidth-bound reasoning, applied to the largest tensor in a serving system.

## Worked Solutions

1. Each element needs only a few floating-point operations but must be read from and written to HBM, so the time to move the bytes dominates the time to compute on them.
2. The next layer consumes the updated residual stream. Without the stored `h`, it must be recomputed from `x` and `res`, adding back the reads the fusion removed.
3. Width 64 is a power of two, so `BLOCK == n_cols` and dividing by either gives the same answer. Widths 100 and 1000 make `BLOCK` larger than `n_cols`, exposing a divide-by-`BLOCK` error.
4. Caches can absorb some of the eager version's repeated reads, so eager traffic is below 10 elements per element; and launch overhead, rows too few to fill the GPU, or a kernel that does not reach peak bandwidth all reduce the gap.
5. Squaring `bfloat16` values and summing across a wide row loses precision in the 8-bit mantissa. Accumulating in `float32` keeps the statistic accurate, and the result is cast back to `bfloat16` only on the store.

## Additional Exercises

6. In `--extras` experiment B, hidden 2560 wastes 37.5% of lanes. Propose a change to the kernel so that it wastes fewer, and say what it costs.
7. Add a test that would have failed the "weight indexed by row offset" mutant at one row but passes it at two rows. Is there one? Explain.
8. The residual add and the norm are fused, but the previous layer's output projection is not. Estimate the extra traffic that fusing the add into that projection's epilogue would remove, for hidden 8192 and 4096 tokens in `bfloat16`.
9. The bf16 residual test accepts either rounding mode. Write the stricter test you would run only on a GPU.
10. `--bench` compares against a `float32` eager reference. Why can that overstate the speedup, and what would you compare against instead?

## Additional Solutions

6. Process the row in two chunks of 2048 and 512 (or loop over fixed 2048-wide tiles), accumulating the sum of squares across tiles, then a second pass to normalise. It wastes almost no lanes but reads `h` twice or holds it in registers, so it trades lane waste for extra traffic or register pressure. Measure before choosing.
7. At one row, `off = row * stride_row + cols` and `cols` coincide for row 0, so a weight loaded at `off` is indistinguishable from one loaded at `cols`. The suite uses rows 1, 2 and 7 and caught it at 15/20, because rows above 0 differ. A one-row-only suite would miss it, which is why the shape sweep includes more than one row.
8. The add currently reads the projection output and the residual and writes the sum, so fusing it into the projection epilogue removes one write and one read of the projection output: 2 x 4096 x 8192 x 2 bytes = about 134 MB, on top of the 268 MB the fused norm already moves. Whether the epilogue fusion is practical depends on the projection kernel.
9. On a GPU, assert `torch.equal(new_res, (x.float() + r.float()).to(torch.bfloat16))`, which holds under round-to-nearest-even and fails in the interpreter, which truncates.
10. The reference upcasts and launches several kernels, so it is slower than a well-tuned eager `bfloat16` RMSNorm module. Compare against the module and the add as your framework actually runs them, ideally including a `torch.compile` version, before quoting a speedup.

## Files and Sources

- `examples/references/part8/ch23_residual_rmsnorm.py`: kernel, `main()`, `--test` (20 self-tests), `--extras` (experiments A to D), `--bench` (GPU only, untested).
- Run with `TRITON_INTERPRET=1` on CPU. Triton 3.9.0 and torch 2.15 were used for the outputs shown.
- Hardware numbers in 23.5 (3.35 TB/s) are the H100 SXM datasheet figure and are used only as a ceiling.
