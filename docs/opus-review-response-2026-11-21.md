# Independent Opus review: response and current status

The external review covers commit ced4863. Its critical runtime-control-flow finding
was correct. It also found genuine numerical edge cases, stale explanations and
overstated evidence claims missed by the earlier self-review. The [original report](https://github.com/MaximumBeings/triton-from-first-principles/blob/main/audit/opus-review-2026-11-21/independent-review.md)
is preserved. This response supersedes the earlier “already fixed” scope statements.

## Critical and numerical corrections

- **Ch1:** Runtime scalar conditions and loop bounds are supported. Static vector extents such as arange bounds are a separate requirement. Added a scalar-control-flow trace and value/alignment-specialization explanation.
- **Ch1/5/9:** Removed the vector-mask “any true” interpretation and block-wide lockstep guarantees. The interpreter’s permissive vector condition need not produce an observable wrong stored result.
- **Ch13:** The historical FP16-output comparison conflates accumulation error with storage rounding. It no longer supports an accumulation-only 20× conclusion; an isolated FP32-output experiment remains unexecuted.
- **Ch14:** Corrected online softmax for an all-negative-infinity leading chunk. Wholly masked rows return zeros by explicit policy, and its arithmetic is upcast to FP32. **Fresh execution pending.**
- **Ch20:** Upcast normalization inputs/affine values to FP32, normalize contiguous wrapper storage and use 16-wide physical dot dimensions. **Fresh execution pending.**
- **Ch21:** Updated physical dot extents, version-specific constraints and scope. Retitled to Forward Attention and Top-2 MoE Gating; full training/advanced features are not claimed. **Fresh execution pending.**
- **Ch22:** Qualified the discount-error dependence (other variables must be controlled), corrected probability/variance wording, added a fixed-draw FP64 payoff oracle assertion and tail-accuracy limits. **The new assertions need execution.**

The [source-transition manifest](https://github.com/MaximumBeings/triton-from-first-principles/blob/main/tools/review_source_revisions.json)
pins the original captured and current hashes. **Twelve complete chapter files need
fresh interpreter and GPU validation**, plus the separate descriptor migration example.
One other file has only the previously registered print-label correction. None of
these later changes is relabelled as having passed its archived H100 run.

## API, teaching and installation corrections

Corrected stale compilation, alignment-hint, masked-memory, register-pressure,
invalid-access and direct-forward-call statements; fixed the contradictory
triton_op/fake-registration explanation. Updated cross-references and the Ch12
figure equation. The direct forward(ctx,x) signature is checked by a plain-Python
regression test, not presented as newly run PyTorch autograd.

[Versions and migration](version-and-migration.md) separates the pinned 3.5.1 recipe,
the 3.3.0 H100 evidence, author-reported historical versions and the removed 3.9
block-pointer API. Added a complete descriptor-copy migration file, clearly marked
unexecuted. Installation now specifies Linux, the GPU capability floor, the chosen
R570+ driver target and Python 3.12 for documentation. R570 is the recipe’s JIT/PTX
target, not a denial of NVIDIA’s distinct CUDA minor-version-compatibility rules.

Added explicit state-reset, autotune-key/search-policy and device-event/rate-calculation
guidance. These are teaching additions, not invented timing captures. Broader kernels
(scans, atomics, persistent/TMA/warp-specialized implementations, LayerNorm backward,
full MoE dispatch and finance extensions) are stated as absent rather than silently
promised. API warnings and limits are scoped by version.

## Evidence correction

The 84-file rerun recorded **zero exits**, not 84 asserted numerical passes. The GPU
was visible, so device queries, factories and timing helpers took different branches.
The [differences report](rerun-differences-2026-11-21.md) links concrete raw logs,
including false first-call slowdown checks and unexpectedly successful timing calls.
Historical page output blocks are now labelled illustrative/possibly abridged;
the timing placeholder template is replaced by an exact captured rerun log.

The published smoke manifest is a derived index, not the collector’s original
byte-identical format. Cold-cache compilation, isolated performance, Ubuntu release
and verbatim equality of all historical page outputs were not established. The
smoke source’s stale module docstring is corrected through an exact metadata-only
source transition; its tested numerical operations and raw logs remain unchanged.

We retain the original system log rather than redact serials/hostname in place:
changing a raw capture would break its published identity. It is a record of a
rented test host, not a secret or independent hardware attestation. Future captures
can use a documented publication/redaction policy with separate raw/published hashes.

## Checks and next execution pass

Offline checks verify 84 current complete listings and matching downloads, migration
source syntax, 13 CPU arithmetic/signature tests, source-transition/evidence integrity,
55 SVGs, JavaScript syntax, a strict site build and internal links. They do **not**
execute the corrected Triton or PyTorch kernels or independently authenticate a host.
The new arithmetic regressions include the all-masked-leading-chunk case and the
FP16-wide-sum overflow boundary.

Next: CPU-only interpreter rerun with CUDA_VISIBLE_DEVICES="" before imports, explicit
output assertions and expected-failure classification; then separate compiled GPU
and sanitizer tests with interpreter mode unset. The full book is not declared
GPU-validated by twelve small smoke cases. A new host is needed for those remaining
execution checks; no GPU execution occurred during this repair pass.
