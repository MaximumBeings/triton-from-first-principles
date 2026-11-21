# Versions, compatibility and migration

Use **Linux, Python 3.12, PyTorch 2.9.1/cu128 and Triton 3.5.1** for the pinned
chapter recipe. This is an installation target, not a newly executed stack.
The real H100 captures used PyTorch 2.7.0/Triton 3.3.0; older illustrated chapter
outputs report 2.13.0/3.7.1. Exact warnings and compiler behavior differ.
Do not combine those transcripts into a single claimed environment.

| API or record | Version-specific meaning |
| --- | --- |
| H100 smoke and 84-file capture | Actual provider stack: torch 2.7.0, Triton 3.3.0; see raw records. |
| Chapter install recipe | torch 2.9.1, Triton 3.5.1; fresh installation/runtime validation pending. |
| Legacy block pointers | Present in 3.3.0/3.5.1. The quoted later deprecation warning is not from those runs. |
| Triton 3.9 | make_block_ptr/advance are removed and raise NotImplementedError; do not upgrade the legacy examples unchanged. |
| NVIDIA non-INT8 dot in 3.5.1 | Backend minimum extents (M,N,K)=(1,1,16); the physical layout may pad smaller M/N. |
| Older 3.3.1 backend cited in historical notes | Minimum extents (16,16,16); this is not a universal rule for newer releases. |

Primary source: [3.5.1 dot constraints](https://github.com/triton-lang/triton/blob/v3.5.1/third_party/nvidia/backend/compiler.py),
[3.9 removed block-pointer API](https://github.com/triton-lang/triton/blob/v3.9.0/python/triton/language/core.py).

## Descriptor migration: a complete example

[Download descriptor_copy.py](examples/migrations/descriptor_copy.py).
It creates a contiguous 16×16 FP32 CUDA input, builds a host TensorDescriptor,
loads a tile and checks all 256 copied values. **GPU execution is pending**;
no result below is presented as captured output.

```bash
unset TRITON_INTERPRET
python docs/examples/migrations/descriptor_copy.py
```

The base and row-stride alignment, contiguous final dimension, physical block shape,
target support and installed API all matter. Host descriptor construction and
device-side make_tensor_descriptor have different setup paths; the latter also
needs the documented descriptor allocator. Do not replace arbitrary strided
block-pointer code by changing only a function name.

## Explicit learning scope

The attention chapter teaches single-head FP32 forward attention and top-2 routing.
It does not implement training gradients, multi-head batching, a Tensor Core
low-precision path, sorting/dispatch/grouped expert GEMM, or a full production MoE
layer. Atomics, scans, persistent kernels, TMA and warp specialization require
additional lessons and hardware validation; they are not promised by its title.

For GEMM, FP32 storage alone does not select IEEE input precision: the examples
request input_precision="ieee". Advanced grouping for L2 reuse and the accumulator
argument to tl.dot are useful extensions, not features measured in the current
small chapter experiments.
