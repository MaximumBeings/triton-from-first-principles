# Interpreter rerun differences — November 21, 2026

The captured runner set CUDA_VISIBLE_DEVICES=1 while also setting TRITON_INTERPRET=1.
The H100 was visible. Interpreter mode changes how Triton kernel bodies execute;
it does not hide CUDA from PyTorch/device queries or guarantee that GPU-dependent
timing helpers fail. The 84 zero exits were not comparisons against the historical
page transcripts. We should have reviewed the differing branches before summarizing them.

The raw logs are retained exactly. Some factory demos created CUDA tensors, hardware
queries returned real H100 properties, autotune/do_bench started with a driver present,
and some first-call slowdown booleans were false. None of these interpreter timings
are GPU-kernel benchmarks. Warmup behavior is not a portable true/false assertion.

## Concrete captured differences

### Log 26: 03_device_placement.py

Page: `docs/part1/03-gpu-side-construction.md`. Recorded exit: `0`.

```text
=== Section 8.3: this environment cannot even construct a CUDA tensor ===
torch.cuda.is_available(): True
created cuda tensor: tensor([1., 2.], device='cuda:0')
```

[Untouched full log](validation/2026-11-21-h100/interpreter/26/output.log) and [command/date/hash index](validation/2026-11-21-h100/interpreter/results.json).

### Log 32: 01_hardware_queries.py

Page: `docs/part1/05-targeting-hardware.md`. Recorded exit: `0`.

```text
=== Section 10.1: querying hardware info without a GPU ===
triton.runtime.driver.active: <nvidia.CudaDriver object at 0x74cab0a43170>
current target: GPUTarget(backend='cuda', arch=90, warp_size=32)
torch.cuda.is_available(): True
_CudaDeviceProperties(name='NVIDIA H100 80GB HBM3', major=9, minor=0, total_memory=81079MB, multi_processor_count=132, uuid=ecb5a396-cc5d-20cb-e3d1-ef08deaa995b, L2_cache_size=50MB)
```

[Untouched full log](validation/2026-11-21-h100/interpreter/32/output.log) and [command/date/hash index](validation/2026-11-21-h100/interpreter/results.json).

### Log 65: 02_autotune_attempt.py

Page: `docs/part5/01-autotuning.md`. Recorded exit: `0`.

```text
ran without error (unexpected in this environment)
```

[Untouched full log](validation/2026-11-21-h100/interpreter/65/output.log) and [command/date/hash index](validation/2026-11-21-h100/interpreter/results.json).

### Log 67: 04_standalone_complete.py

Page: `docs/part5/01-autotuning.md`. Recorded exit: `0`.

```text
ran without error (unexpected in this environment)
```

[Untouched full log](validation/2026-11-21-h100/interpreter/67/output.log) and [command/date/hash index](validation/2026-11-21-h100/interpreter/results.json).

### Log 68: 01_do_bench_attempt.py

Page: `docs/part5/02-benchmarking-correctly.md`. Recorded exit: `0`.

```text
=== Section 19.1: attempting triton.testing.do_bench itself ===
ran without error (unexpected in this environment)
```

[Untouched full log](validation/2026-11-21-h100/interpreter/68/output.log) and [command/date/hash index](validation/2026-11-21-h100/interpreter/results.json).

### Log 69: 02_warmup_and_statistics.py

Page: `docs/part5/02-benchmarking-correctly.md`. Recorded exit: `0`.

```text
first call vs. later calls -- first is measurably slower: False
```

[Untouched full log](validation/2026-11-21-h100/interpreter/69/output.log) and [command/date/hash index](validation/2026-11-21-h100/interpreter/results.json).

### Log 71: 04_standalone_complete.py

Page: `docs/part5/02-benchmarking-correctly.md`. Recorded exit: `0`.

```text
=== Section 19.1: attempting triton.testing.do_bench itself ===
ran without error (unexpected)
first call slower than steady-state average? False
BLOCK=256 is slower under this interpreter? True
 not real GPU occupancy -- never extrapolate this to real hardware)
```

[Untouched full log](validation/2026-11-21-h100/interpreter/71/output.log) and [command/date/hash index](validation/2026-11-21-h100/interpreter/results.json).

## What a corrective rerun must do

Run the CPU-only interpreter pass with CUDA_VISIBLE_DEVICES="" set before imports,
record torch.cuda.is_available(), use exact source hashes, and classify deliberate
negative examples separately. Compare outputs with asserted numerical oracles and
explicit expected outcomes, rather than only process exit codes. Capture every
failed/unexpected assertion and keep warnings. Run the compiled GPU suite separately
with interpreter mode unset. Current corrected chapter sources and the descriptor
migration example still need these reruns; no such new run is claimed here.

The published smoke manifest is a **derived subset/index** of two commands from the
larger retained collector, not its byte-identical original output format. The source
collector wrote a list including H100-book commands; publication selected the system
and smoke records and added source metadata. Raw command logs and their hashes remain
unchanged. Cache state was not captured; this is not a cold-compilation measurement.
The captures overlap other activity on GPU 1, so no isolated performance claim is valid.
The raw system record establishes Linux/kernel/glibc, not an explicit Ubuntu release:
/etc/os-release was not collected by that command. Future collectors should include it.
