# "Triton from First Principles" — Independent Review (Opus, 4 passes)

Site: https://maximumbeings.github.io/triton-from-first-principles/ · repo `github.com/MaximumBeings/triton-from-first-principles` · reviewed at commit `ced4863` · Nov 21 2026
Scope: all 22 chapters (Parts 0–7), Appendices A–D (incl. all 22 quiz answers), the five top-level pages, `audit/`, the H100 validation records, `tools/`, install instructions, site build and the image-viewer overlay. Review only; nothing in the repo was changed.

Sandbox limits (stated so you know what "verified" means here): no GPU, no PyTorch (download.pytorch.org is blocked and the CUDA wheel does not fit the disk). Kernels were run through Triton's interpreter on numpy-backed stand-ins for `torch`, and compiled to cubin for sm_90 **without a GPU** (Triton 3.9) to test what the real compiler accepts. Triton source for v3.3.1 / 3.5.1 / 3.7.1 and PyTorch v2.9.1 was read directly.

## Bottom line

| Severity | Count |
|---|---|
| CRITICAL | 1 |
| MODERATE | 29 |
| MINOR | 48 |
| NIT | 18 |

1. **The numbers in the book are real.** Every printed number reviewers could reproduce (strides, sector counts, matmul results, softmax, Black–Scholes, Monte-Carlo, all 22 quiz answers) came out right. No fabricated results were found.
2. **The H100 evidence looks authentic but proves less than it appears to.** GPU UUID, SM count, clocks, driver/CUDA/torch/Triton versions, Triton 3.3.0 warning line numbers, run durations and all 91 hashes agree with each other. It shows 12 small fp32 smoke cases passing on one H100 and 84 sources exiting 0 under the interpreter. It does **not** show that the printed outputs are correct, that any chapter kernel works on the compiled GPU path, or anything about backward kernels, autotuning or performance. The repo's `verify_*.py` / `test_*.py` tools only check self-consistency (hashes, AST parse, hard-coded re-derivations); none could detect fabricated evidence.
3. **The "complete interpreter rerun" contradicts several chapters and nobody noticed.** It ran with the H100 still visible (`CUDA_VISIBLE_DEVICES=1`), so demos the chapters describe as "no GPU here" took other paths (a CUDA tensor was created; `autotune` and `do_bench` ran "unexpectedly"; the first-call warmup slowdown was **False**). All were counted as passes because only exit codes were recorded. The review pages never mention this.
4. **"Genuinely captured output" is often not verbatim.** The chapters use that phrase 94 times; only 24 of 84 logs appear verbatim in their pages. Others are reformatted, contain hand-typed placeholders such as `[varies]ms`, or show text no code prints. This is the hand-edited-output pattern from the earlier books.
5. **The "already fixed" notes in the self-review pages are only partly true.** Stale pre-fix text survives in at least eight places (Ch2 L458, Ch7 Worked Solution 3, Ch9 §9.2, Ch11 summary, Ch20 OOB explanation, Ch22 discount-factor lesson and its payoff check, Appendix C opening), contradicting the corrected text elsewhere in the same chapter.
6. **Many launch settings and a headline API do not work on the stated target.** Chapter 20's `BLOCK_K=4` and Chapter 21's 4-wide tiles fail to compile for H100; Chapter 7's `tl.make_block_ptr` raises `NotImplementedError` on current Triton 3.9.0 (block pointers were removed).
7. **The version story is inconsistent**: install target torch 2.9.1 / Triton 3.5.1; printed outputs authored on 2.13.0 / 3.7.1; the H100 run used 2.7.0 / 3.3.0; current release is 2.15.0 / 3.9.0. Readers cannot reproduce version-specific output.

## Top findings to fix first

| ID | Where | Problem |
|---|---|---|
| **T2-Ch01-1 (CRITICAL)** | Ch1 §1.2 table L121 | Says runtime scalars cannot drive `if`/`for` in a kernel. False: `for i in range(0, n)` and `if n > 3` with runtime `n` lower to `scf.for` / `scf.if` (TTIR checked). Later K-loops depend on the opposite being true. |
| T2-Ch01-2 | Ch1/Ch6 | Never teaches value/alignment specialization (int == 1 → constexpr; ints divisible by 16 and 16-byte-aligned pointers get divisibility attributes). So "changing a runtime arg never recompiles" and "strides are just integers" are wrong; `n=16→17` or `stride=1→4` recompiles. |
| T2-Ch07-1 | Ch7 §7.3 | Block-pointer API: deprecation warning quoted exists only in 3.7/3.8 (not 3.3.0, 3.5.1 target, or 3.9); on 3.9.0 `make_block_ptr` raises `NotImplementedError` and files 03/04 crash. |
| T1-EV-1/2 | evidence pages | Interpreter rerun ran with the GPU visible; contradicts Ch8/Ch10/Ch18/Ch19 printed claims; Ch19's "genuine, repeatable warmup effect" is disproved by log 69 (`first is measurably slower: False`). |
| T1-EV-3 | all chapters | 94 uses of "genuinely captured"; only 24/84 verbatim; some blocks hold literal placeholders. |
| T3-Ch13-1 | Ch13 §13.3 | fp32-vs-fp16 accumulation comparison is confounded: the "fp32 accumulator" kernel stores fp16, so the ~9.7e-5 is output rounding (fp16 rounding of the fp64 truth gives the same 2.3255e-4). True accumulation gap ≈ 8000×, not "~20×". |
| T3-Ch14-1 | Ch14 §14.3 | `online_softmax_kernel` returns NaN for the whole row if the first chunk is all −inf (masked/causal case) — and it is presented as the FlashAttention mechanism. |
| T4-Ch20-2 | Ch20 | LayerNorm never converts fp16 input to fp32; `tl.sum` does not promote; a 4096-wide row with mean 20 sums to 81,788 > 65,504 → inf → NaN (compiled IR shows `addf … f16`). |
| T4-Ch20-1 / T4-Ch21-2 | Ch20, Ch21 | `BLOCK_K=4` and 4-wide tiles fail `tl.dot` minimum-size rules on 3.5.1 (K ≥ 16) and 3.9 (K ≥ 8); Ch21's warning cites the wrong version and rule. |
| T4-Ch22-2 | Ch22 §22.1.2 | The chapter's own six contracts contradict "error grows with r·T" (T=2 contract has smaller error than the T=1 ATM one). The claim is true other things equal (sweep: −3.07% at T=0.01 … −47.93% at T=5). |
| T3-Ch18-1 / Ch19-2 | Ch18, Ch19 | "autotune cannot run under the interpreter" / "every boolean reproduces reliably — verified" are contradicted by the repo's own logs 65, 67, 68, 69, 71. |

## Batch T1 — top-level pages, evidence, install, build, Appendices A & B

**H100 evidence verdict:** authentic-looking, limited in what it proves (see Bottom line 2). Cross-file consistency checks that passed: GPU UUID in interpreter log 32 equals bus 07:00.0 in `system.log` (`CUDA_VISIBLE_DEVICES=1`); H100 SXM5 specs (132 SMs, 50 MB L2, 81559 MiB, 1980/2619 MHz, 700 W, 18 NVLinks; HBM3 3.35 TB/s recomputed); driver 580.105.08 / nvcc 12.8.93 / torch 2.7.0 / triton 3.3.0 mutually consistent (torch 2.7.0 pins triton 3.3.0 on PyPI); warnings at `interpreter.py:453/:544` match v3.3.0 source lines; run durations fit workloads; all 91 `checksums.json` entries, both manifest log hashes, 84 source hashes (against commit c9fb730) and 84 log hashes verify; logs 7 and 36 reproduced byte-for-byte; Black–Scholes relative differences recomputed.

| ID | Sev | Finding |
|---|---|---|
| T1-EV-1 | MODERATE | Rerun left `CUDA_VISIBLE_DEVICES=1`; logs 26, 32, 65, 68 diverge from chapters ("cannot construct a CUDA tensor", "0 active drivers"); `execution-2026-11-21.md` says "They used CPU tensors/interpreter semantics" which log 26 contradicts. Fix: rerun with `CUDA_VISIBLE_DEVICES=""`, list diverging files. |
| T1-EV-2 | MODERATE | `part5/02-benchmarking-correctly.md:108,126` "genuine, repeatable warmup effect" disproved by logs 69, 71. |
| T1-EV-3 | MODERATE | "genuinely captured" blocks not verbatim (e.g. `part0/03…md:100-101`, `part4/01…md:98-110`, `part5/02…md:526,529` placeholders). |
| T1-INST-1 | MODERATE | Three version stacks (authoring 2.13/3.7.1; target 2.9.1/3.5.1; H100 run 2.7.0/3.3.0); current 2.15.0/3.9.0. Ch7 "genuine, current warning from Triton 3.7" absent from 3.3.0 log and 3.5.1/3.9 source. |
| T1-EV-4 | MINOR | "Exact capture manifest" is an edited excerpt (script writes a list incl. rows from another project; published file is a dict with keys the script never writes). |
| T1-EV-5 | MINOR (PLAUSIBLE) | Smoke process 1.64 s total, 0.55 s for ~10 kernel specialisations → Triton cache likely warm → earlier unrecorded runs; "first compiled GPU smoke run" should be softened. |
| T1-EV-6 | MINOR | `README.md:13` and `gpu_smoke.py:1` still say the smoke test has not been run. |
| T1-EV-7 | MINOR | Interpreter rerun and smoke run overlapped on GPU 1 (13:20:54); "Ubuntu 24.04" only inferred from glibc 2.39. |
| T1-INST-2/3/4 | MINOR | No compute-capability (≥ 8.0) requirement; WSL not a listed Triton platform; driver minimum for cu128 not stated; docs build needs Python ≥ 3.11.5 (fails on Ubuntu 22.04's 3.10). |
| T1-IMG-1, T1-A11Y-1 | MINOR | Generic alt text on appendix figures reused as dialog `aria-label`; viewer fit-toggle changes label *and* sets `aria-pressed`; error `<p>` lacks `role="alert"`; clicking dialog padding closes it. |
| T1-EV-8 | NIT | Public `system.log` includes GPU serials and IP-style hostname — consider redacting. |
| T1-INST-5, IMG-2, QUIZ | NIT | Getting Started §6 omits `verify_execution.py`; grammar slip in Appendix A; `--site` should be `--site-dir`; "CHAPTER 00" label on homepage figure; every SVG embeds ~47 KB font + licence (2.7 MB total); quiz says it follows chapter numbers but cites none. |

**Spot-checks:** 22 specific self-review claims checked: 17 true, 2 partly true, 3 false (README smoke status, "exact" manifest, reproducibility target). Site build: `mkdocs build --strict` passes; link check: 32 pages, 12,252 targets, no duplicate IDs; nav matches the 31 `.md` files (no orphans).
**Appendix B:** all 22 quiz answers verified correct.
**Smoke kernel** looks sound (exact fraction inputs vs fp64 reference, guard cells, `input_precision="ieee"`).

## Batch T2 — Chapters 1–11 (Parts 0–1)

All 40 reference files ran under the interpreter on Triton 3.9.0 and 3.5.1; compiled-frontend behaviour checked via TTIR (sm_90). Every numeric Expected Output reproduces except the Ch7 block-pointer files on 3.9.

**Chapter 1** — **T2-Ch01-1 CRITICAL** (runtime scalars and control flow, above). **Ch01-2 MODERATE** specialization rules missing. **Ch01-3 MODERATE** fp16 accumulation explanation wrong: ulp near 0.2 is 2^-13 (not 2^-11); the sum comes out high (0.2338) not low because once the accumulator reaches 0.0625 each 0.0001 step rounds *up* to a full ulp (0.000061 for 512 steps, then 0.000122 for 891); "eleven additional mantissa bits" is 13. **Ch01-4 MODERATE** `if mask:` — interpreter treats any multi-element tensor as always true (`bool(data) if data.size == 1 else True`, interpreter.py:1020-1024), not "at least one True"; the printed "silently wrong" output is identical to the correct `tl.where` output (compiled mode correctly errors). Repeated in Ch5 §5.1 and Ch9 Solution 2. Ch01-5 MINOR: failure to call without `[grid]` is a fixed `RuntimeError("Cannot call @triton.jit'd outside…")`, rationale speculative. Ch01-6 NIT: "argument argument types"; "CPython never compiles".

**Chapter 2** — Ch02-1 MODERATE: line 458 says the extra `COMPILING … BLOCK = 4` line is "genuine compilation — exactly the cache-keyed behavior", contradicting line 53/Solution 1 (stale pre-fix text). Ch02-2 MINOR: key semantics incomplete (dtypes are part of the key; "runtime cost zero" ignores lookup).

**Chapter 3** — Ch03-1 MINOR: inline "Genuinely captured" transcripts not printed by any shipped file (values right). Ch03-2 NIT: 28 B stride — one in eight adjacent thread pairs share a segment; 32 B is NVIDIA's actual sector size, not "simplified". Verified: 28 sectors/896 B/57.1% AoS; 16 sectors SoA; 4 vs 32 segments.

**Chapter 4** — Ch04-1 MINOR: `tl.sum` is partly sequential (per-thread partials, warp shuffles, shared memory). Verified IDs, cdiv, matmul.

**Chapter 5** — **Ch05-1 MODERATE**: within-program "lockstep" model is wrong — a program's lanes are spread over `num_warps` warps with several elements per thread; warps run independently (even threads within a warp are not guaranteed lockstep since Volta); Triton gives block-value semantics. Ch05-2 MINOR: "grid-stride loop" misnomer; "summation is associative" false for floats.

**Chapter 6** — Ch06-1 MINOR: pointer args are `!tt.ptr<f32>`, not "indistinguishable from an integer"; `[527455040]` is interpreter handle data. Ch06-2 MINOR: wrapper with `BLOCK = next_power_of_2(n)` and grid (1,) fails above 2^20 elements (`numel exceeds triton maximum tensor numel (1048576)`), taught as canonical.

**Chapter 7** — **Ch07-1 MODERATE** block-pointer API (above). **Ch07-2 MODERATE** Worked Solution 3 says `multiple_of(offsets,16)` asserts *every* value is a multiple of 16 — contradicts §7.2 and the docstring ("first element of every contiguous group"); stale pre-fix answer. Ch07-3 MINOR: promises `tl.advance` in Ch8 (appears only in Appendix D.3). Ch07-4 NIT.

**Chapter 8** — Ch08-1 MINOR: Triton's launcher itself raises `ValueError("Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)")`; pinned host memory passes (PLAUSIBLE). Ch08-2 MINOR: "this environment cannot construct a CUDA tensor" vs rerun logs 26/32 (GPU visible; H100/132 SMs printed) — the real device-mismatch error and populated `device_caches` could have been shown.

**Chapter 9** — Ch09-1 MINOR: intuition/table say masked lanes still issue loads; contradicts corrected Solution 3. Ch09-2 NIT: hardcoded column stride 1 / row-major output pointers (the habit Ch3 warns against).

**Chapter 10** — Ch10-1 MINOR (PLAUSIBLE): `num_warps` can change reduction combine order → bitwise float differences. Ch10-2 NIT: no actual hardware numbers; `x*2.0` labelled "tensor-core-style".

**Chapter 11** — Ch11-1 MINOR: summary says BLOCK "directly controls real per-thread register usage" — contradicts §11.2/Solution 2 (stale). Ch11-2 MINOR: `device_caches` transcript abridged but labelled "Genuinely captured"; probe tests `.cache`, absent on both classes, so proves nothing. Ch11-3 NIT: "149 public attributes" matches neither 3.5.1 (145) nor 3.9 (153).

**Cross-cutting:** X-1 MODERATE version inconsistency. X-2 MINOR broken forward references: Ch6 "Chapter 20 on quantization" (none), Ch8 fill fusion "(Chapter 20)", Ch2 "(Chapter 10)" host-reference lifetime, Ch5 "Chapter 11 returns to races", Ch3 "Chapter 13 third broadcast dimension" — none exist where cited.
SVGs ch01–ch11 arithmetic all correct. Counts: 1 CRITICAL / 8 MODERATE / 15 MINOR / 6 NIT.

## Batch T3 — Chapters 12–19, Appendices C & D

Counts: 0 CRITICAL / 9 MODERATE / 13 MINOR / 4 NIT.

- **Ch12** — MINOR: promises Ch19 will measure a fusion effect (it measures no GPU timing). NIT: SVG says `2x+1`, text `2x−1`; torch called "trusted oracle" but it is 5e-7 off fp64 truth (Triton 5e-9). Verified: GELU values bit-for-bit; tanh-approx error 4.12e-4 (book 4.13e-4).
- **Ch13** — **MODERATE T3-Ch13-1** (confounded accumulation comparison, above). MINOR: "tl.dot on fp16 *often* gives fp32 result" — default `out_dtype=float32` always; cites v3.3.1 min dot size (16,16,16) but target 3.5.1 is (1,1,16) so BLOCK_M=N=4 legal but BLOCK_K=4/8 not (3.9 compile fails "K >= 8"); never mentions fp32 `tl.dot` defaults to TF32 (code hardcodes `ieee`); no GROUP_SIZE_M L2 ordering, no `acc = tl.dot(a,b,acc)` idiom. NIT: "max abs diff" lines not printed by shown file. Verified: 8×8 diff 1.5259e-05, 5×7×6 diff 3.81e-06, sentinel 2994004.0.
- **Ch14** — **MODERATE T3-Ch14-1** (online softmax NaN when first chunk all −inf; fix e.g. `alpha = tl.where(m_new == -inf, 0, tl.exp(m - m_new))`). MINOR: ASCII diagram exponents wrong (exp(−51) is 7.1e-23 not 1.9e-23; exp(−101) 1.4e-44 not 1.7e-44; e^101 7.3e43); WE 14.3.1 output contains "single-pass online softmax" that the code never prints and contradicts the chapter's own two-pass argument (hand-edited); no cross-program reductions (atomics, two-pass, `tl.reduce`), no fp16→fp32 upcast in softmax.
- **Ch15** — **MODERATE**: §15.2 trap and Solution 3 say `TritonGELU.forward(x)` "often appears to work … no grad_fn"; with `forward(ctx, x)` it raises `TypeError: forward() missing 1 required positional argument: 'x'` (plain Python semantics; not run — no torch). MINOR coverage: `needs_input_grad`, `setup_context`, `mark_non_differentiable`, `once_differentiable`, `gradcheck`. GELU derivative finite-difference check over [−8,8]: max err 7.3e-10.
- **Ch16** — **MODERATE**: WE 16.2.2 calls missing-mask overrun "a real, reproducible crash"/"illegal memory access"; small overruns usually don't fault (caching allocator rounds to 512 B, 2 MiB segments) — silent read/corruption is typical; interpreter did not crash in 3 runs. Verified: erf-on-fp16 rejected in compiled mode too; cdiv(10000,1024)=10 programs, 784 valid lanes.
- **Ch17** — NIT: WE 17.1.2 tensors not printed by the file (consistent: correct grads exactly 2× buggy). Verified `next_functions`, reverse call order, fan-in summation.
- **Ch18** — **MODERATE T3-Ch18-1**: "autotune cannot run at all under `TRITON_INTERPRET=1`" holds only when no GPU driver is present; with a driver (logs 65/67) it benchmarks interpreted CPU executions and caches a winner on exactly the misleading signal Ch19 warns about. MINOR: key semantics omit that every tensor arg's dtype is appended (3.5.1 `autotuner.py:216-219`); no disk caching, `prune_configs_by`, deprecated `warmup/rep`, `TRITON_PRINT_AUTOTUNING`. Verified traceback line numbers match 3.7.1.
- **Ch19** — **MODERATE** `do_bench` "defaults toward robust statistics": default is `return_mode="mean"` (testing.py:127; the chapter's own table says so; Triton's docstring misleadingly says median). **MODERATE** "reproduces reliably — verified directly" contradicted by logs 68, 69, 71. MINOR: "outlier ten times larger than the rest combined" (it is ~10× one trial; others sum 5.88 s); no `torch.cuda.Event` example, `do_bench_cudagraph`, launch-overhead discussion, clock locking, or bandwidth/TFLOPs calculation. Verified do_bench mechanics vs source lines 147–190; 294.79/38.82 = 7.59×.
- **Appendix C** — **MODERATE** opening of §C.2 says a direct Triton launch "is invisible to torch.compile" — contradicts L24, L81, its own SVG and the PyTorch tutorial (user-defined Triton kernels supported). **MODERATE** Outcomes 1–3 teach `register_fake` "is what Dynamo's tracing needs" for `triton_op`; `triton_op` already registers the fn as its fake (`result.register_fake(fn)`, v2.9.1 `torch/_library/triton.py`), observed failures were artifacts of CPU tensors + interpreter; re-registering overrides. MINOR: no `register_autograd` example; snippets never revalidated (appendix admits).
- **Appendix D** — MINOR: `make_block_ptr` deprecation UserWarning exists only in 3.7.1. Verified alignment messages/docstring semantics.

## Batch T4 — Chapters 20–22

Every printed number reproduced independently (PyTorch's CPU RNG rebuilt in numpy; `randn` outputs match exactly). All 12 page listings byte-identical to downloads. Counts: 0 CRITICAL / 8 MODERATE / 11 MINOR / 4 NIT.

**Chapter 20 — Fused Kernels** (5 MODERATE)
- Ch20-1: `fused_linear_gelu(BLOCK_K=4, …)` (L370, L617) cannot compile on any GPU Triton (3.3.1/3.5.1 need K ≥ 16; 3.9 K ≥ 8); BK=8/16 compile; no tile-size warning in this chapter.
- Ch20-2: LayerNorm fp16 overflow (above).
- Ch20-3 (PLAUSIBLE): wrapper does `empty_like(x)` but uses `x.stride(0)` as output row stride and assumes column stride 1 — non-contiguous or transposed `x` writes wrong rows/past the buffer. Fix: pass output strides or `.contiguous()`.
- Ch20-4: OOB-probe output block (L105–111) is in a format PyTorch cannot print (it switches to scientific notation when values span >1000×; never abbreviates a 20-element tensor); bias 0..15 isn't the section's bias — edited, labelled "genuinely captured".
- Ch20-5: `final-review-2026-11-21.md` says "Removed guarantees that out-of-bounds reads silently succeed", but L113, trap L134 and Solution 2 still explain silent success as a property of the read; Ch16 says the same read crashed in the same interpreter.
- MINOR: "sqrt(eps) ~140× larger than eps" (L277) should be 316 (L288/292 say 316; measured ~283 because row variance 1.36e-12); "genuinely new launch pattern" (L208) false — one-program-per-row used in Ch14:34; worked-example outputs from unshown code (values reproduced: 0.43509950 vs 0.43509936, variances match); homepage promises "Fused LayerNorm/softmax" but Part 6 has no fused softmax, RMSNorm or LayerNorm backward.
- NIT: `weight.t().contiguous()` is an extra copy per call (use strides); leftover "“Two touches total”" phrasing; "this chapter's foundation" should be Chapter 21.
- Verified: epilogue math, erf GELU, `SQRT_2_INV: tl.constexpr`, biased variance with eps inside sqrt, N/(N−1) ratio, padded-lane masking, SVG and quiz arithmetic.

**Chapter 21 — Flash Attention, MoE, Advanced Features** (1 MODERATE)
- Ch21-1: title promises "Advanced Features" — no `tl.associative_scan`, atomics, `tl.inline_asm_elementwise`, persistent kernels, TMA or warp specialization. The attention kernel is single-head, fp32, forward-only: no backward (dQ/dK/dV), no log-sum-exp, no batch/heads, no fp16/bf16 tensor-core path, no exp2/log2e scaling (cf. Triton tutorial 06). MoE: top-2 gating only (no sorting/padding, grouped GEMM, load balancing; chapter says so). Fix: retitle or add.
- MINOR: tile-size note cites v3.3.1 and "≥16 in M/N/K"; on target 3.5.1 M,N may be 1 and only K ≥ 16 — failing dims are both K dims (`BLOCK_D=8`, `BLOCK_N=4`); inconsistent cross-reference ("Chapter 13.2's sentinel trick" vs "Chapter 14's reduction-sentinel pattern"). NIT: diagram says diagonal block is "the only block needing tl.where", code applies it on every block.
- Verified: online-softmax recurrences (FA2-style); causal bound `min(N, max_col)`; numpy port matches naive attention (non-causal 1.5e-7, causal 2.0e-7, several block shapes, no NaN from padding); MoE top-1 [4,5,7,5,6,5], top-2 [5,6,3,4,7,1], sums and mean exact; `tl.topk` exists in 3.5.1/3.9.

**Chapter 22 — Quantitative Finance** (2 MODERATE)
- Ch22-1: learning objective L14 and L203 promise `torch.allclose` payoff checks against a CPU implementation on fixed draws — no such code exists; final-review lists it fixed.
- Ch22-2: discount-factor lesson contradicted by the chapter's data (above). The final review says "error/maturity attribution" was corrected — only the printed label was.
- MINOR: Background table names the closed-form *kernel* as the oracle; code uses scipy/numpy. fp32 `0.5*(1+erf(d/√2))` cancels for deep OTM (emulated fp32, S=100, T=0.5, r=0.05, σ=0.2: K=170 → 1.5e-3 relative error; K=200 → 56%); use `0.5*erfc(-d/√2)` or fp64. Some outputs from unshown code (values reproduced). Scope: normals come from host `torch.randn`, no in-kernel Philox/`tl.randn` anywhere in the book, no variance reduction, Greeks/puts, trees or VaR/CVaR; "Monte Carlo kernel" only maps pre-drawn normals to payoffs. NIT: "Chapter 1's earliest lessons — masking, oracle verification" (Ch1 is types/JIT).
- Verified: d1/d2/price formulas; N(d1)=delta, N(d2)=risk-neutral exercise probability; error formula K·N(d2)·(1−e^{−rT}); put–call parity residual ≤ 7e-15; wrong-discount relative errors [−26.12, −9.96, −19.95, −16.95, −17.93, −3.07]; rolling vol 8 values and biased variants exact (pandas defaults to ddof=1); MC 10.405878 ± 0.032836 (1.3615 SE from BS); undiscounted 10.939399 ± 0.034520 (14.16 SE, 4.68% off); standard error correctly discounted.

## What I could not verify
- Any compiled-GPU execution of the chapter kernels (no GPU here; the compile-to-cubin checks show the compiler accepts/rejects them, not that they run correctly).
- Anything that needs real PyTorch (autograd.Function behaviour at runtime, `torch.compile` integration, allocator behaviour); those findings are marked PLAUSIBLE where applicable.
- That the H100 run happened (consistent and physically plausible, but self-reported with self-written manifests).

## Recommended order of fixes
1. T2-Ch01-1 and Ch01-2 (foundation errors), then T3-Ch13-1, T3-Ch14-1, T4-Ch20-2/3, T4-Ch22-2.
2. Make Ch7's block-pointer chapter version-accurate (3.9 removes the API; show `tl.make_tensor_descriptor`), and fix tile sizes (Ch20 `BLOCK_K ≥ 16`, Ch21 warning).
3. Re-run the interpreter pass with `CUDA_VISIBLE_DEVICES=""` and compare each log to the printed output; either paste raw logs or relabel blocks "abridged"; remove placeholders.
4. Sweep stale pre-fix text (Ch2 L458, Ch7 Solution 3, Ch9 §9.2, Ch11 summary, Ch20 OOB, Ch22 payoff check/discount lesson, Appendix C opening).
5. Pick one version stack and state each transcript's provenance on Getting Started; update install requirements (compute capability ≥ 8.0, driver ≥ R570 for cu128, Python ≥ 3.11.5 for the docs build).
6. Fix cross-references that point to chapters lacking the cited content; retitle Ch21 or add the missing material.
7. Update README/`gpu_smoke.py` smoke-test status; mark the capture manifest an excerpt; redact serials/hostname from `system.log`.
