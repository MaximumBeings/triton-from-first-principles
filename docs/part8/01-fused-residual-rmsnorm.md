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

## Chapter Summary

RMSNorm does a few operations per element, so it is limited by HBM bandwidth, and its floor is bytes moved divided by bandwidth. Fusing the residual add into the normalization cuts element transfers from about 10 to 4 per element, a predicted ceiling of 2.5x on traffic, provided the kernel also writes the updated residual. The kernel accumulates in `float32`, divides by the true row width rather than the block width, and keeps `eps` inside the square root; it matched a reference on power-of-two and awkward widths and on a near-zero row in the interpreter. Speed was not measured: the benchmark and Nsight Compute procedure are given for you to run on an H100.

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
