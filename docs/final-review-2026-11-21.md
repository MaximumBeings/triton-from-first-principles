# Final adversarial review — November 21, 2026

Three independent reviewers challenged the foundation chapters and installation,
the advanced kernels and worked answers, and the illustrations and teaching flow.
A follow-up checked the corrections. This is an offline technical review, not a
new GPU run or a proof that every possible input is supported.

!!! warning "Superseded in part by the independent review"
    This page records the earlier review pass. The independent review identified missed errors, stale claims and new numerical edge cases. Its [response and current pending status](opus-review-response-2026-11-21.md) take precedence; the older source-scope claims below are historical.

## Confirmed issues corrected

| Area | Correction |
| --- | --- |
| JIT evidence | Interpreter dtype messages no longer claim observed GPU compilations or cache entries. |
| Invalid accesses | Removed guarantees that out-of-bounds reads silently succeed; they may fault, corrupt data or appear successful. |
| Memory accounting | Distinguished the AoS model's 28-sector union footprint from 112 sector requests across four field instructions; actual DRAM traffic needs measurement. |
| Masking | False memory-access lanes do not perform loads/stores; whole-block skipping additionally avoids instructions and address work. |
| Alignment hints | Explained contiguous groups and group-start divisibility in Chapter 7 and Appendix D. |
| Floating point | FP32 accumulation also rounds; reduction order, K and inputs affect error. Online-softmax identities are exact in real arithmetic, not rounding-free implementations. |
| Autograd | Added the first wrappers' small contiguous-FP32 contract and distinguished derivative support from graph capture. |
| Autotuning | Added repeated-execution and reset/restore guidance, a mutable-output walkthrough, and compiled-candidate correctness checks. Interpreter equality is scoped to the shown elementwise example. |
| LayerNorm and attention | Scoped the one-program-row design; attention query tiles maintain independent state rather than synchronizing programs. |
| MoE | Moved the normalized-top2 policy contract before the lesson; other weight policies are not universally wrong. |
| Finance | Corrected N(d1)/N(d2), error/maturity attribution and sample-variance versus standard-deviation terminology; distinguished deterministic fixed-draw checks from statistical price comparisons. |
| PyTorch integration | Supported direct Triton launches can be captured by torch.compile; registration is an additional integration route rather than a universal prerequisite. |
| Memory appendix | Separated register pressure from spilling, qualified dot lowering/staging, added descriptor prerequisites and scoped version-specific warnings. |
| Teaching flow | Removed an absent dropout promise, corrected cross-references, made Mojo comparisons optional and moved the long homepage walkthrough after the introduction. |
| Illustrations | Corrected the gradient banner's forward/derivative notation; added four appendix banners and four explanatory SVGs. All 55 figures have embedded fonts and checked labels. |

Primary references used to resolve the disputed contracts:
[Triton debugging](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html),
[NVIDIA memory accesses](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html),
[alignment](https://triton-lang.org/main/python-api/generated/triton.language.multiple_of.html),
[contiguity](https://triton-lang.org/main/python-api/generated/triton.language.max_contiguous.html),
[autotuning](https://triton-lang.org/main/python-api/generated/triton.autotune.html),
[descriptors](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html),
[PyTorch direct-launch integration](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html),
and the [Black–Scholes derivation](https://book.derivative-securities.org/Chapter_BlackScholes.html).

## Evidence preserved and remaining validation

No numerical kernel operations changed in this review. One complete finance file
changes only a misleading printed label: the largest relative error in its example
belongs to the first T=1 contract, not the longest T=2 contract. The archived source
and raw log remain exact. The [source-transition manifest](https://github.com/MaximumBeings/triton-from-first-principles/blob/main/tools/review_source_revisions.json)
pins both hashes; the verifier permits exactly that single print-string replacement
and rejects additional changes. The corrected label itself has not been rerun.

The [November 21 record](execution-2026-11-21.md) still contains 12 passing compiled
H100 smoke cases and 84 zero-exit interpreter captures. Eighty-three captured
reference sources remain byte-identical; one has the label-only revision above.
The appendix snippets were not among those 84 files and have not been rerun on
that host. Full compiled chapter coverage, appendix execution, backward GPU
validation, broader inputs, sanitizer coverage and performance testing remain
separate work. A new GPU is not required for these prose, diagram and label fixes.

## Offline validation

- 84 complete current listings parse and match their downloadable sources.
- Ten independent CPU arithmetic tests cover the worked examples, sector counting and finance attribution.
- Three evidence regression tests reject source drift and edited captured logs.
- All 55 SVGs pass XML, embedded-font, glyph, bounds and label-overlap checks.
- Strict MkDocs build and built internal-link/fragment checks pass.

Historical outputs are not edited to match new wording, and no verifier is disabled
to obtain a successful build. The review found no new critical numerical-kernel
issue within the stated examples; its scope does not certify untested contracts.
