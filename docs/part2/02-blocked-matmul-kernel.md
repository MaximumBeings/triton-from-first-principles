# Chapter 13: The Blocked Matmul Kernel

> "Every kernel through Chapter 12 computed each output position from exactly one corresponding input position — masking kept a launch honest at its edges, but no output ever depended on more than a handful of nearby inputs. A matmul breaks that completely: every single output element depends on an entire row of one matrix and an entire column of the other, and the only way to compute it without reading the whole thing back from memory over and over is to accumulate across tiles, in a loop, inside the kernel itself."

**What you will understand by the end of this chapter:**

- How to structure a genuinely tiled matrix multiplication: a 2D grid of output tiles, and a loop within each program instance that accumulates partial products across the shared dimension, using `tl.dot` for the actual per-tile multiplication
- Why a matmul kernel needs masking on *three* dimensions — `M`, `N`, and `K` — not just the two a 2D elementwise kernel would need, and what happens, concretely and dramatically, when the `K` mask is the one that's missing
- Why accumulating in `fp32` matters more for a matmul's K-loop than it did for Chapter 1's single reduction, because a real matmul's accumulation runs for as many steps as there are `K`-tiles, compounding rounding error at every one
- How this chapter's tiling structure is the direct, larger-scale descendant of Chapter 11's single-tile `tl.dot` example, now looping across as many tiles as the shared dimension actually requires

**What you need to know first:**

- Chapter 11, Section 11.3's `tl.dot` introduction and its `[[19, 22], [43, 50]]` single-tile example — this chapter builds the real, multi-tile version of exactly that kernel
- Chapter 5's masking discipline and Chapter 9's structural masking — both are extended here to a third dimension
- Chapter 1, Section 1.3's fp16-vs-fp32 accumulation finding — Section 13.3 is a direct, deeper continuation of that same lesson

## 13.1 Tiling a Real Matmul: A 2D Grid and a K-Loop `[FOUNDATIONAL]`

### Intuition

Assembling a large jigsaw puzzle that's too big to lay out on one table happens in sections: you build one region at a time, and within each region, you still have to place every piece that belongs there before that region is done — you don't finish a region by placing just one piece from it. Computing one output tile of a matrix product is exactly this: `C`'s tile at rows `[m0:m0+BLOCK_M)` and columns `[n0:n0+BLOCK_N)` depends on *all* of `A`'s corresponding rows and *all* of `B`'s corresponding columns, across the *entire* shared dimension `K` — and if `K` is larger than one block's worth, computing that one output tile correctly requires visiting every `K`-tile and accumulating its contribution before the output tile is finished.

This is the one new structural idea this chapter introduces: a loop, inside the kernel, over chunks of `K`, each iteration loading one tile of `A` and one tile of `B`, multiplying them with `tl.dot`, and adding the result into a running accumulator — precisely the `tl.zeros(...)`-as-accumulator pattern from Chapter 5's reductions, now accumulating matrix tiles instead of scalars.

### Background

| | An elementwise kernel (Ch. 12) | This chapter's matmul kernel |
|---|---|---|
| Grid shape | 1D, one program instance per block of output positions | 2D — `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`, one program instance per output *tile* |
| Does computing one output position require a loop? | No — one masked load, one masked store | **Yes — a loop over `K` in chunks of `BLOCK_K`, accumulating each tile's contribution** |
| What accumulates across loop iterations | Nothing — Ch. 12's kernels had no loop at all | A `(BLOCK_M, BLOCK_N)`-shaped `tl.zeros(...)` accumulator, exactly Chapter 5's reduction pattern, now 2D |
| Primitive doing the per-tile work | Elementwise arithmetic (`tl.maximum`, `+`, `*`) | `tl.dot(a_tile, b_tile)` — Chapter 11's primitive, used here in earnest |

### Worked Example 13.1.1 — a matmul requiring a genuine multi-iteration K-loop

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        b_mask = (ks[:, None] < K) & (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
```

Against genuine `(8, 8) @ (8, 8)` matrices, with `BLOCK_M = BLOCK_N = BLOCK_K = 4`, the grid is `(cdiv(8,4), cdiv(8,4)) = (2, 2)` — four program instances, each computing one `4×4` tile of the `8×8` output — and each program instance's `K`-loop runs `cdiv(8, 4) = 2` iterations, accumulating two `4×4×4` partial products before its tile is complete. Genuinely captured:

```
M,K,N = 8 8 8  grid = (2, 2)  K-iterations per tile = 2
matches torch matmul? True
max abs diff: 1.52587890625e-05
```

The tiny remaining difference (`~1.5e-5`) is ordinary floating-point noise from summing the same values in a different order than PyTorch's own matmul does internally — not a bug, and well within the tolerance any matmul comparison should use (Chapter 12, Section 12.3 already established why exact bit-for-bit matches aren't the right bar for floating-point computation).

### ASCII Diagram — one output tile, two K-iterations

```
   A (8x8)                          B (8x8)                    C (8x8), tile (0,0) shown
   +---+---+                        +---+---+                  +----+----+
   |A00|A01|  <- K-tile 0            |B00|B01|                  |C00 | .. |
   +---+---+     (BLOCK_K=4)        +---+---+                  +----+----+
   |A10|A11|                        |B10|B11|                  | .. | .. |
   +---+---+                        +---+---+                  +----+----+

   program (0,0) computes C00:
     acc  = 0
     acc += A00 @ B00     (K-iteration 0: ks = [0,1,2,3])
     acc += A01 @ B10     (K-iteration 1: ks = [4,5,6,7])
     C00  = acc            (2 iterations because K=8, BLOCK_K=4 -> cdiv(8,4)=2)
```

> `[COMMON TRAP]` It's easy to write the accumulator as `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)` *inside* the `for k0 in range(...)` loop instead of before it, by analogy with ordinary Python code where a fresh variable is often declared right where it's first used. Placed inside the loop, the accumulator would be reset to zero on every iteration, discarding every prior `K`-tile's contribution the moment the next one starts — the final result would only ever reflect the *last* `K`-tile, silently, with no error raised. The accumulator has to be initialized exactly once, before the loop begins, precisely like Chapter 5's scalar reduction accumulator.

## 13.2 Masking Every Dimension a Matmul Has: `M`, `N`, and `K` `[FOUNDATIONAL]`

### Intuition

A librarian shelving books by both subject and author still has to know when they've reached the end of the shelf in *both* directions — running out of subjects to file under says nothing about whether there's still another author's worth of books to place. A matmul kernel has exactly this two-and-then-a-third kind of edge: `M` and `N` bound the output tile's position, exactly as Chapters 5 and 9 already established for two-dimensional bounds masks — but `K`, the dimension being summed over, is a third, independent bound that has nothing to do with where the output tile sits, and every `K`-loop iteration needs its own fresh bounds check against it.

### Background

| | `M` / `N` masking | `K` masking |
|---|---|---|
| What it bounds | Which output tile positions are real | Which elements of the *current* `K`-chunk are real |
| How many times it's checked | Once per program instance (computed once, reused for every load/store) | **Once per `K`-loop iteration** — a fresh comparison against `K` every time `ks` advances |
| What happens if it's missing | Chapters 5 and 9's familiar out-of-bounds corruption | The same shape of bug, but compounded by `tl.dot` across every affected `K`-chunk — Worked Example 13.2.2 |

### Worked Example 13.2.1 — correct three-way masking on deliberately awkward dimensions

Against `M=5, K=7, N=6` — none of them a multiple of `BLOCK=4` — `matmul_kernel` from Section 13.1, with its `a_mask = (rows[:, None] < M) & (ks[None, :] < K)` and matching `b_mask`, produces:

```
M,K,N = 5,7,6 (none divisible by BLOCK=4)
grid = (2, 2)  K-iterations per tile = 2
matches torch matmul? True
max abs diff: 3.814697265625e-06
```

Every dimension's mask did its job independently: the output tile's edges (`M`, `N`) and the shared dimension's final, partial chunk (`K`) were all handled correctly by the same three-comparison pattern, with no dimension's correctness depending on either of the others being right.

### Worked Example 13.2.2 — omitting only the `K` mask, made deterministically visible with a sentinel

```python
a_mask = (rows[:, None] < M)          # BUG: no ks < K check
b_mask = (cols[None, :] < N)          # BUG: no ks < K check
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)
```

To make the consequence unmistakable rather than a subtle, allocator-dependent difference, `A` and `B`'s real `(M, K)` and `(K, N)` data are sliced out of a larger underlying buffer whose out-of-range columns/rows are filled with an obvious sentinel, `999.0` — the same technique Chapter 9's ragged-batch example used. Genuinely captured:

```
torch reference (ground truth):
 tensor([[1.2000, 1.3000, 1.4000, 1.5000],
        [3.2000, 3.5500, 3.9000, 4.2500],
        [5.2000, 5.8000, 6.4000, 7.0000],
        [7.2000, 8.0500, 8.9000, 9.7500]])

correct kernel (K masked):
 tensor([[1.2000, 1.3000, 1.4000, 1.5000],
        [3.2000, 3.5500, 3.9000, 4.2500],
        [5.2000, 5.8000, 6.4000, 7.0000],
        [7.2000, 8.0500, 8.9000, 9.7500]])
matches? True

buggy kernel (K NOT masked -- reads the 999.0 sentinel past the real K):
 tensor([[2994004.0000, 2994004.2500, 2994004.5000, 2994004.5000],
        [2994006.2500, 2994006.5000, 2994006.7500, 2994007.2500],
        [2994008.2500, 2994009.0000, 2994009.5000, 2994010.0000],
        [2994010.2500, 2994011.0000, 2994012.0000, 2994012.7500]])
matches? False
```

`999.0 × 999.0 ≈ 998001`, and the last `K`-chunk here contributes three sentinel-times-sentinel terms into the accumulator (`K=5`, `BLOCK_K=4`, so the second `K`-iteration covers indices `[4,5,6,7]`, three of which — `5, 6, 7` — are past the real `K=5` and land entirely on sentinel data in both `A` and `B`) — `3 × 998001 ≈ 2,994,003`, matching the corrupted output almost exactly. This is not a subtle numerical drift; it's the accumulator faithfully summing exactly what it was told to sum, including three tiles' worth of `999 × 999` that should never have been read at all.

### ASCII Diagram — where the missing mask lets sentinel data in

```
  A's real data (K=5)         A's underlying buffer (padded to 8 columns)
  +---+---+---+---+---+       +---+---+---+---+---+---+---+---+
  | real columns 0..4 |       | real columns 0..4 |999|999|999|
  +---+---+---+---+---+       +---+---+---+---+---+---+---+---+
                                                    ^^^^^^^^^^^ sentinel, never meant to be read

  K-loop, BLOCK_K=4:  k0=0 -> ks=[0,1,2,3]  (all real -- fine either way)
                      k0=4 -> ks=[4,5,6,7]  (index 4 real; 5,6,7 are sentinel)

  WITH K-mask:    ks < K=5 -> [T,F,F,F] -- indices 5,6,7 masked out, other=0.0 used instead
  WITHOUT K-mask: no check at all       -- indices 5,6,7 read the raw 999.0 sentinel directly
```

> `[COMMON TRAP]` It's tempting to assume that getting `M` and `N` masking right is "the hard part" of a matmul kernel, since that's the pattern already familiar from Chapters 5 and 9 — and then apply the identical two-dimensional mask pattern to the `a`/`b` loads without separately checking `K`. Worked Example 13.2.2's corruption happened with *correct* `M`/`N` masks in place the whole time; the bug was entirely in the dimension that has nothing to do with output tile position at all. A matmul kernel's masking checklist has one more item on it than a 2D elementwise kernel's does, and it's easy to reuse a mental model that only accounts for two.

## 13.3 Accumulation Precision at Real Depth: Why `fp32` Accumulation Matters More in a Loop `[FOUNDATIONAL]`

### Intuition

Chapter 1, Section 1.3 showed that summing 2,048 copies of a small `fp16` number one at a time in `fp16` drifted about 14% from the exact answer, while accumulating the identical values in `fp32` stayed within a fraction of a percent — a single reduction, over many terms. A matmul's `K`-loop is the same accumulation pattern, except every "term" is itself a full tile's worth of dot products, and the number of accumulation *steps* scales with how many `K`-tiles exist — for a realistically deep `K`, that can easily be tens or hundreds of iterations, each one an opportunity for the accumulator's precision to matter.

### Background

| | `fp32` accumulator (`matmul_kernel_fp32_acc`) | `fp16` accumulator (`matmul_kernel_fp16_acc`) |
|---|---|---|
| Accumulator dtype across the `K`-loop | `tl.float32`, regardless of input dtype | `tl.float16` — rounds after every single `K`-iteration |
| When precision is lost | Only once, casting the final result down to `fp16` for storage | Repeatedly — once per `K`-iteration, compounding |
| Deviation from an `fp64` ground truth, `K=1024` (128 iterations of `BLOCK_K=8`) | `~9.7e-5` | **`~2.0e-3` — roughly 20× worse** |
| Where this matters most | Any real matmul with a nontrivial shared dimension — which is most of them | — |

### Worked Example 13.3.1 — the same K-loop, two accumulator dtypes, a deep `K`

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # fp32 accumulator
for k0 in range(0, K, BLOCK_K):
    ...
    acc += tl.dot(a, b)
...
tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)   # cast down only at the very end
```

against

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)   # fp16 accumulator
for k0 in range(0, K, BLOCK_K):
    ...
    partial = tl.dot(a, b).to(tl.float16)
    acc += partial   # rounds to fp16 every single K-chunk
```

With `fp16`-stored inputs, `K=1024`, `BLOCK_K=8` (128 `K`-iterations per output tile), compared against an `fp64` "ground truth" computed from the identical `fp16`-rounded inputs — isolating accumulation error from the unavoidable, one-time rounding cost of storing the inputs as `fp16` in the first place — genuinely captured:

```
K=1024, BLOCK_K=8 -> 128 K-iterations
fp32acc max abs diff vs exact: 9.737179152580211e-05
fp16acc max abs diff vs exact: 0.001974425897060428
```

The `fp32`-accumulated kernel stays within `~1e-4` of the `fp64` ground truth even after 128 accumulation steps; the `fp16`-accumulated kernel drifts roughly twenty times further, for exactly the reason Chapter 1, Section 1.3 identified: `fp16` has too little mantissa precision to keep absorbing small updates onto a growing running total without repeatedly rounding away part of each addition.

> `[COMMON TRAP]` `tl.dot(a, b)` on `fp16` inputs often produces an `fp32`-typed result already, on real hardware, as part of how tensor-core matmul instructions work — but that only protects the *result of one `tl.dot` call*. Worked Example 13.3.1's buggy version explicitly cast that per-tile `fp32` result back down to `fp16` before adding it into the accumulator (`tl.dot(a, b).to(tl.float16)`), which throws away exactly the precision `tl.dot` had just provided. The accumulator's own declared dtype — not merely `tl.dot`'s internal behavior — is what determines whether that precision survives across the whole `K`-loop or gets discarded one `K`-tile at a time.

## 13.4 Complete Runnable Code

Sections 13.1–13.3 built up this chapter's findings as isolated probes. The files below are the actual, complete, run-in-order source. `04_standalone_complete.py` combines everything from Sections 13.1 through 13.3 into a single script; its output is reproduced in full afterward. All were run exactly as shown, with `TRITON_INTERPRET=1` set, per Chapter 1's opening disclosure.

### File: `01_tiled_matmul.py` — Section 13.1

**Execution:** `TRITON_INTERPRET=1 python3 01_tiled_matmul.py`

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        b_mask = (ks[:, None] < K) & (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a, b, BLOCK_M=4, BLOCK_N=4, BLOCK_K=4):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


def demo_tiled_matmul():
    print("=== Section 13.1: a real multi-tile matmul, requiring a K-loop and a 2D grid ===")
    M, K, N = 8, 8, 8
    A = torch.arange(M * K, dtype=torch.float32).reshape(M, K) * 0.1
    B = torch.arange(K * N, dtype=torch.float32).reshape(K, N) * 0.1
    C = triton_matmul(A, B, BLOCK_M=4, BLOCK_N=4, BLOCK_K=4)
    ref = A @ B
    print("M,K,N =", M, K, N, " grid =", (triton.cdiv(M, 4), triton.cdiv(N, 4)),
          " K-iterations per tile =", triton.cdiv(K, 4))
    print("matches torch matmul?", torch.allclose(C, ref, atol=1e-3))


if __name__ == "__main__":
    demo_tiled_matmul()
```

### File: `02_three_way_masking.py` — Section 13.2

**Execution:** `TRITON_INTERPRET=1 python3 02_three_way_masking.py`

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        b_mask = (ks[:, None] < K) & (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def matmul_kernel_no_k_mask(a_ptr, b_ptr, c_ptr, M, N, K,
                             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M)
        b_mask = (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(kernel, a, b, K_logical, BLOCK_M=4, BLOCK_N=4, BLOCK_K=4):
    M = a.shape[0]
    N = b.shape[1]
    c = torch.empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](a, b, c, M, N, K_logical,
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


def demo_correct_awkward_dims():
    print("=== Section 13.2: correct three-way masking, awkward dimensions ===")
    M, K, N = 5, 7, 6
    A = torch.arange(M * K, dtype=torch.float32).reshape(M, K) * 0.1
    B = torch.arange(K * N, dtype=torch.float32).reshape(K, N) * 0.1
    C = run(matmul_kernel, A, B, K)
    ref = A @ B
    print(f"M,K,N = {M},{K},{N} (none divisible by BLOCK=4)")
    print("matches torch matmul?", torch.allclose(C, ref, atol=1e-3))


def demo_missing_k_mask():
    print("\n=== Section 13.2: omitting the K mask, made visible with a sentinel ===")
    M, K_real, N = 4, 5, 4
    SENTINEL = 999.0
    A_padded = torch.full((M, 8), SENTINEL)
    A_padded[:, :K_real] = torch.arange(M * K_real, dtype=torch.float32).reshape(M, K_real) * 0.1
    A = A_padded[:, :K_real]

    B_padded = torch.full((8, N), SENTINEL)
    B_padded[:K_real, :] = torch.arange(K_real * N, dtype=torch.float32).reshape(K_real, N) * 0.1
    B = B_padded[:K_real, :]

    ref = A @ B
    C_correct = run(matmul_kernel, A, B, K_real)
    C_buggy = run(matmul_kernel_no_k_mask, A, B, K_real)

    print("correct kernel matches?", torch.allclose(C_correct, ref, atol=1e-4))
    print("buggy kernel matches?  ", torch.allclose(C_buggy, ref, atol=1e-4))
    print("buggy kernel result (should be dramatically wrong):\n", C_buggy)


if __name__ == "__main__":
    demo_correct_awkward_dims()
    demo_missing_k_mask()
```

### File: `03_accumulation_precision.py` — Section 13.3

**Execution:** `TRITON_INTERPRET=1 python3 03_accumulation_precision.py`

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_fp32_acc(a_ptr, b_ptr, c_ptr, M, N, K,
                            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        b_mask = (ks[:, None] < K) & (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


@triton.jit
def matmul_kernel_fp16_acc(a_ptr, b_ptr, c_ptr, M, N, K,
                            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        b_mask = (ks[:, None] < K) & (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        partial = tl.dot(a, b).to(tl.float16)
        acc += partial
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(kernel, a, b, dtype_out, BLOCK_M=4, BLOCK_N=4, BLOCK_K=8):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=dtype_out)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](a, b, c, M, N, K,
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


def demo_accumulation_precision():
    print("=== Section 13.3: fp32 vs fp16 accumulation over a deep K-loop ===")
    torch.manual_seed(0)
    M, K, N = 4, 1024, 4
    A = (torch.randn(M, K) * 0.1).to(torch.float16)
    B = (torch.randn(K, N) * 0.1).to(torch.float16)
    exact = A.double() @ B.double()

    out_fp32acc = run(matmul_kernel_fp32_acc, A, B, torch.float16)
    out_fp16acc = run(matmul_kernel_fp16_acc, A, B, torch.float16)

    print(f"K={K}, BLOCK_K=8 -> {triton.cdiv(K, 8)} K-iterations")
    print("fp32acc max abs diff vs exact:", (out_fp32acc.double() - exact).abs().max().item())
    print("fp16acc max abs diff vs exact:", (out_fp16acc.double() - exact).abs().max().item())


if __name__ == "__main__":
    demo_accumulation_precision()
```

### File: `04_standalone_complete.py` — the full chapter, one program

**Execution:** `TRITON_INTERPRET=1 python3 04_standalone_complete.py`

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        b_mask = (ks[:, None] < K) & (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def matmul_kernel_no_k_mask(a_ptr, b_ptr, c_ptr, M, N, K,
                             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M)
        b_mask = (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def matmul_kernel_fp16_acc(a_ptr, b_ptr, c_ptr, M, N, K,
                            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        b_mask = (ks[:, None] < K) & (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        partial = tl.dot(a, b).to(tl.float16)
        acc += partial
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(kernel, a, b, dtype_out, BLOCK_M=4, BLOCK_N=4, BLOCK_K=4):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=dtype_out)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](a, b, c, M, N, K,
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


def main():
    print("=== Section 13.1: tiled matmul, 2x2 grid, 2 K-iterations ===")
    M, K, N = 8, 8, 8
    A = torch.arange(M * K, dtype=torch.float32).reshape(M, K) * 0.1
    B = torch.arange(K * N, dtype=torch.float32).reshape(K, N) * 0.1
    C = run(matmul_kernel, A, B, torch.float32)
    print("matches torch?", torch.allclose(C, A @ B, atol=1e-3))

    print("\n=== Section 13.2: sentinel-based K-mask trap ===")
    M2, K_real, N2 = 4, 5, 4
    SENTINEL = 999.0
    A_padded = torch.full((M2, 8), SENTINEL)
    A_padded[:, :K_real] = torch.arange(M2 * K_real, dtype=torch.float32).reshape(M2, K_real) * 0.1
    Ap = A_padded[:, :K_real]
    B_padded = torch.full((8, N2), SENTINEL)
    B_padded[:K_real, :] = torch.arange(K_real * N2, dtype=torch.float32).reshape(K_real, N2) * 0.1
    Bp = B_padded[:K_real, :]
    ref2 = Ap @ Bp
    C_correct = run(matmul_kernel, Ap, Bp, torch.float32)
    C_buggy = run(matmul_kernel_no_k_mask, Ap, Bp, torch.float32)
    print("correct kernel matches?", torch.allclose(C_correct, ref2, atol=1e-4))
    print("buggy kernel matches?  ", torch.allclose(C_buggy, ref2, atol=1e-4))

    print("\n=== Section 13.3: fp32 vs fp16 accumulation ===")
    torch.manual_seed(0)
    M3, K3, N3 = 4, 1024, 4
    A3 = (torch.randn(M3, K3) * 0.1).to(torch.float16)
    B3 = (torch.randn(K3, N3) * 0.1).to(torch.float16)
    exact = A3.double() @ B3.double()
    out_fp16acc = run(matmul_kernel_fp16_acc, A3, B3, torch.float16, BLOCK_K=8)
    print("fp16acc max abs diff vs exact:", (out_fp16acc.double() - exact).abs().max().item())


if __name__ == "__main__":
    main()
```

### Expected Output for `04_standalone_complete.py`

```
=== Section 13.1: tiled matmul, 2x2 grid, 2 K-iterations ===
matches torch? True

=== Section 13.2: sentinel-based K-mask trap ===
correct kernel matches? True
buggy kernel matches?   False

=== Section 13.3: fp32 vs fp16 accumulation ===
fp16acc max abs diff vs exact: 0.001974425897060428
```

Every claim here was verified with full numeric detail earlier in this chapter: the exact tile-by-tile hand trace in Section 13.1, the sentinel-derived `~2,994,003` corruption in Section 13.2, and the full `fp32`-versus-`fp16` comparison against an `fp64` ground truth in Section 13.3.

## Chapter Summary

A tiled matmul kernel needs exactly one genuinely new structural idea beyond what Chapters 1 through 12 already established: a loop over the shared dimension `K`, accumulating each tile's `tl.dot` product into a running total initialized once, before the loop, using the same accumulator pattern Chapter 5's reductions introduced (Section 13.1). That loop needs its own bounds check — `K` masking is a third, independent requirement alongside the familiar `M`/`N` output-tile masking, and Section 13.2 showed both a correct three-way-masked matmul on deliberately awkward dimensions and, with a deterministic sentinel-based construction, exactly how dramatically wrong the result becomes when only the `K` mask is missing. And accumulation precision, first introduced in Chapter 1 as a single reduction over 2,048 terms, matters even more inside a matmul's `K`-loop, where every additional `K`-tile is another accumulation step — Section 13.3 showed a `20`× gap between `fp32` and `fp16` accumulation open up over 128 iterations, far larger than the single-reduction gap Chapter 1 first measured.

## Self-Check Questions

1. Why does a matmul kernel need a loop inside each program instance, when every kernel through Chapter 12 computed its output with a single masked load and store?
2. What specifically goes wrong if the accumulator `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)` is placed inside the `K`-loop instead of before it?
3. A matmul kernel correctly masks `rows[:, None] < M` and `cols[None, :] < N` on every load and store, but never checks `ks < K` anywhere. Will this kernel produce a compile error, a runtime error, or a silently wrong result? Explain why, referencing Worked Example 13.2.2.
4. Worked Example 13.3.1 found a `~20`× larger deviation from the `fp64` ground truth for the `fp16`-accumulated kernel compared to the `fp32`-accumulated one. Why does this gap grow with the number of `K`-iterations, rather than staying constant?
5. `tl.dot(a, b)` on `fp16` inputs can produce an `fp32`-typed result even before you do anything else to it. Why doesn't this alone guarantee that a matmul kernel's overall accumulation is accurate?

## Where We Go Next

The matmul kernel in this chapter accumulates a genuine reduction — a sum over `K` — but hides that reduction entirely inside `tl.dot`, which computes it as a single opaque operation. Chapter 14 turns to reductions written out explicitly: sums, maxes, and the numerically trickier softmax, where the reduction itself, not a black-box primitive, is the entire content of the kernel, and where the same accumulation-precision lessons from this chapter reappear in a form you have to manage by hand rather than delegate to `tl.dot`.

## Worked Solutions

**1.** Every kernel through Chapter 12 computed each output position from a small, fixed set of nearby inputs — one masked load per output position was always enough. A matmul's output position depends on an *entire* row of one matrix and an *entire* column of the other, and when that shared dimension `K` is larger than a single block can hold, the only way to cover all of it is to process it in chunks, one `BLOCK_K`-sized tile at a time, accumulating each chunk's contribution — which requires a loop, since there is no single masked load wide enough to cover an arbitrarily large `K` in one step.

**2.** Placing `acc = tl.zeros(...)` inside the loop resets the accumulator to zero at the start of every iteration, which discards every previously accumulated `K`-tile's contribution the instant the next iteration begins. The final stored result would reflect only the *last* `K`-tile processed, as if `K` were equal to `BLOCK_K` regardless of its real value — silently wrong, with no error raised, exactly the same shape of mistake as forgetting to initialize a scalar accumulator before a reduction loop (Chapter 5).

**3.** It produces a silently wrong result, not a compile or runtime error — exactly as Worked Example 13.2.2 demonstrated. `ks` is an ordinary runtime-computed offsets tensor; there's nothing in the language that requires it to be checked against `K` before being used in a `tl.load`, any more than any other unchecked pointer arithmetic in this book (Chapter 6). For any `K`-loop iteration where the current `K`-chunk extends past the real `K`, the unmasked load reads whatever data happens to lie past the tensor's real extent — Worked Example 13.2.2's sentinel construction made this land on a deliberately obvious `999.0`, producing an unmistakably wrong `~2,994,003`-magnitude result instead of a subtle drift.

**4.** The `fp16` accumulator rounds its running total back to `fp16` precision after *every single `K`-iteration*, so each of the 128 iterations in Worked Example 13.3.1 is an independent opportunity to lose a small amount of precision, and these losses compound across iterations rather than canceling out — this is exactly Chapter 1, Section 1.3's finding about repeated rounding of a running total, now happening once per `K`-tile instead of once per element. The `fp32` accumulator only loses precision once, when the final result is cast down to `fp16` for storage, so its error stays roughly constant regardless of how many `K`-iterations occurred, while the `fp16` accumulator's error grows with the loop's length.

**5.** `tl.dot(a, b)`'s `fp32`-typed result only reflects the precision of that *one* tile's multiplication. What happens to that result immediately afterward is a separate decision entirely: Worked Example 13.3.1's buggy kernel explicitly cast that `fp32` result back down to `fp16` (`tl.dot(a, b).to(tl.float16)`) before adding it into an `fp16` accumulator, discarding the precision `tl.dot` had just provided before the addition even happened. The accumulator's own declared dtype across the whole `K`-loop — not `tl.dot`'s internal per-call behavior — is what determines whether that precision survives accumulation or gets thrown away one `K`-tile at a time.
