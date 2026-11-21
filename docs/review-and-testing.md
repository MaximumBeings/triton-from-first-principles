# Technical Review and Testing Status

**Review date: November 21, 2026.** This pass reviewed technical contracts, source listings, installation instructions and diagrams without a CUDA host. A subsequent H100 session passed 12 compiled smoke cases and reran all 84 complete references in interpreter mode. This does not establish full compiled-GPU coverage; see the [dated execution record](execution-2026-11-21.md).

See also the [final adversarial review and corrections](final-review-2026-11-21.md).

## What changed

| Area | Correction |
| --- | --- |
| Homepage | Replaced the obsolete “only Chapter 1 exists” notice; listed the actual 22 chapters and four appendices; removed promises of absent grouped expert GEMM, bond/credit/duration implementations and universal autograd coverage. |
| JIT/cache | Pointer arguments do not automatically specialize on tensor shape. Corrected the claimed meaning of interpreter `static_print` messages and distinguished specialization keys from final instruction equality. |
| GPU resources | BLOCK is a logical tile extent, not a register count in one thread. Corrected the BLOCK=65536 explanation, staging-knob guarantees and classic-language versus experimental API scope. |
| Matrix operations | Marked hand-sized dot tiles as interpreter-specific. FP32 teaching dots now explicitly request IEEE input precision; their previous interpreter outputs are not GPU results. |
| Attention | Replaced loop-continue control flow with a bounded causal loop, documented physical tile limits and top-left masking, and qualified numerical, tie, all-masked-row and MoE gating contracts. |
| Top-k | Removed the blanket assertion that Triton has no built-in top-k; linked the modern API and distinguished this hand-built value/index selection. |
| Autograd | Flat-address GELU wrappers now make inputs/upstream gradients contiguous. Documented first-order scope and the need for a separate higher-order implementation. |
| Finance | Distinguished N(d1) from risk-neutral exercise probability N(d2), added domain/zero-boundary requirements, and explained variance and Monte Carlo conventions. |
| Installation | Selected a documented PyTorch 2.9.1/cu128 + Triton 3.5.1 target, addressed ARM64 separately, preserved provider drivers and separated documentation dependencies from GPU dependencies. This target has not been GPU-tested in this pass. |
| Reader experience | Added 55 original embedded-font SVGs (22 chapter banners, two entry-page banners, four appendix banners and 27 worked/overview figures), chapter-level download links, a worked quiz and a same-page viewer with fit-width/fit-whole controls and navigation cleanup. |

## Offline checks and their limits

All **84 explicitly titled complete Python reference listings** parse and have byte-identical downloads with recorded hashes. The new GPU-smoke source also passes Python syntax checks. Syntax parsing does not import PyTorch/Triton, compile a GPU kernel or execute its mathematics.

Ten CPU binary64 tests check the worked stride/mask/GEMM/GELU/softmax/variance/option/payoff arithmetic. SVG checks examine embedded fonts, glyph coverage, label bounds and overlap. A strict documentation build and built-link check cover page structure and targets. These checks are not a substitute for interpreter reruns, GPU numerical tests, sanitizer checks or benchmarking.

The [original source-identity inventory](https://github.com/MaximumBeings/triton-from-first-principles/blob/main/audit/original-reference-identities.json) records the pre-review baseline. Historical outputs already printed in the chapters remain author-reported interpreter results. The repository did not supply their original raw logs, package snapshots or per-run timestamps. This review does not invent that historical evidence. A separate interpreter rerun records the captured source hashes and fresh logs. A later print-label-only correction in one finance file is explicitly pinned; its revised label has not been recaptured. No numerical operations changed. Current downloads identify current source; a historical printed result is not a new validation of its hash.

## First GPU validation and remaining work

The complete [new NVIDIA smoke source](examples/gpu_smoke.py) was compiled and executed on H100. It allocates CUDA tensors, uses physically valid dot tiles and checks twelve cases: three masked strided additions, three ragged GEMMs and six attention cases against CPU references. All twelve cases passed on November 21. The [raw captured output](execution-2026-11-21.md) records the actual provider stack and source identity; twelve smoke cases are not the full chapter suite.

An H100 or GH200 is suitable for this single-GPU pass. GH200 needs the matching ARM64 software stack. Ordinary chapter references often allocate CPU tensors or deliberately demonstrate invalid kernels; do not bulk-run them in GPU mode by only unsetting TRITON_INTERPRET. Port and classify each test's contract, including intentional failures, before claiming compiled coverage.

A future capture must include exact source hashes, commands, real CPU/GPU/driver and package versions, UTC execution dates, exit codes and unedited stdout/stderr. Measure performance only on the real GPU and state timing scope. No numerical outputs or timings from a future GPU run are fabricated here.

## Primary references

- [Triton interpreter and debugging limits](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- [Triton dot precision and operand contracts](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
- [Pinned NVIDIA v3.3.1 dot shape contract](https://github.com/triton-lang/triton/blob/v3.3.1/third_party/nvidia/backend/compiler.py) and [semantic checks](https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/language/semantic.py)
- [Pinned frontend visitor implementation](https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/compiler/code_generator.py)
- [Modern top-k API](https://triton-lang.org/main/python-api/generated/triton.language.topk.html)
- [pandas rolling standard deviation convention](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.Rolling.std.html)
