# H100 execution — November 21, 2026

The recorded compiled GPU smoke session for this edition used one GPU on a **two-H100
80GB HBM3 host**. Linux kernel 6.8.0-1064-nvidia, glibc 2.39, x86-64, driver 580.105.08, Python 3.12.3,
PyTorch **2.7.0**, Triton **3.3.0**, PyTorch CUDA runtime **12.8**.
These are the provider's actual packages; the separately documented 2.9.1/3.5.1
installation target was not installed or validated by this run.

## Compiled GPU smoke test

Source commit: `c9fb730eb42ba407e5b3398c62ed4830706477ca`.
Source SHA-256: `af64ff8cd0901ddc33bac19c2e9ced721fac5f48c886b1b9d9ff50056a941165`.
Start UTC: `2026-11-21T13:20:54.640652+00:00`. Finish UTC: `2026-11-21T13:20:56.276257+00:00`.
Exit code: **0**. The captured source snapshot retains its original pre-run “pending” docstring. The downloadable current smoke file changes only that docstring; its exact metadata-only transition is checked separately. This record applies to the captured hash.

```bash
CUDA_VISIBLE_DEVICES=1 python3 docs/examples/gpu_smoke.py
```

Interpreter mode was unset. This compiled real GPU kernels and checked all output
elements against CPU oracles for three strided additions, three ragged GEMMs
and six attention cases. It did not compile all 84 chapter files for the GPU,
run a sanitizer, validate backward kernels on the GPU or measure performance.

```text
{
  "started_utc": "2026-11-21T13:20:55.723495+00:00",
  "platform": "Linux-6.8.0-1064-nvidia-x86_64-with-glibc2.39",
  "machine": "x86_64",
  "python": "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]",
  "torch": "2.7.0",
  "triton": "3.3.0",
  "torch_cuda": "12.8",
  "gpu": "NVIDIA H100 80GB HBM3",
  "capability": [
    9,
    0
  ],
  "scope": "compiled GPU correctness; no performance measurement"
}
PASS strided-add N=1; complete output and both guards
PASS strided-add N=137; complete output and both guards
PASS strided-add N=4099; complete output and both guards
PASS ragged-gemm M=17 N=23 K=19; complete CPU FP64 oracle
PASS ragged-gemm M=16 N=16 K=16; complete CPU FP64 oracle
PASS ragged-gemm M=1 N=7 K=33; complete CPU FP64 oracle
PASS attention M=17 N=23 D=8 causal=False; CPU FP64 oracle
PASS attention M=17 N=23 D=8 causal=True; CPU FP64 oracle
PASS attention M=20 N=20 D=8 causal=False; CPU FP64 oracle
PASS attention M=20 N=20 D=8 causal=True; CPU FP64 oracle
PASS attention M=3 N=19 D=16 causal=False; CPU FP64 oracle
PASS attention M=3 N=19 D=16 causal=True; CPU FP64 oracle
GPU SMOKE PASS: 12 compiled GPU cases; not the complete chapter suite
```

[Raw output](validation/2026-11-21-h100/triton-smoke.log),
[derived capture index](validation/2026-11-21-h100/manifest.json),
[tested source snapshot](validation/2026-11-21-h100/gpu_smoke.py),
[full system capture](validation/2026-11-21-h100/system.log).

## Complete chapter references: interpreter rerun

All **84 complete reference files at the captured commit exited zero** under
`TRITON_INTERPRET=1` on the same host. They ran with interpreter semantics and a visible GPU; some device/factory queries therefore took CUDA branches.
this count is not 84 GPU-compiled programs. Several files intentionally print
negative comparisons to demonstrate bugs. A zero process exit is not proof of
all numerical paths. Full source hashes, commands, UTC start/finish times, exit
codes and unchanged logs are in the [84-file results index](validation/2026-11-21-h100/interpreter/results.json).

The runner used four independent processes with CPU math thread counts set to
one; CUDA visibility was GPU 1. Its [complete source](validation/2026-11-21-h100/run_triton_references.py)
is retained. Each result's log path is relative to the `interpreter/` directory.
Historical printed chapter outputs remain historical; these new logs record the
captured source identities without editing those earlier outputs. The independent review subsequently prompted numerical and launch-contract fixes. See the [review response](opus-review-response-2026-11-21.md) for exact changed hashes and pending reruns. The historical 84 zero exits do not validate those later sources. See [observed environment differences](rerun-differences-2026-11-21.md).

## Rerun environment and publication details

See the [differences and collector-format explanation](rerun-differences-2026-11-21.md). The run did not hide the GPU, validate every printed output or establish a cold-cache compile time.

## What remains

Port and test each chapter's device allocations and physical tile shapes before
claiming full compiled-GPU coverage. Backward/gradient correctness, sanitizer
coverage, broader dtypes/layouts, performance and the selected fresh-install
package target still need dedicated validation. No additional rented GPU is
needed to finish publishing this evidence.
