# Triton From First Principles — Under Development

**Building Production GPU Kernels in OpenAI Triton, One Layer at a Time**

!!! info "This book is just getting started"
    This is the sibling project to [*Mojo From First Principles*](https://maximumbeings.github.io/mojo-from-first-principles/), built the same way: one chapter at a time, in page order, each one carrying real runnable code and genuinely captured output rather than illustrative numbers. Only the scaffold and Chapter 1 exist so far — the rest of this page describes where the book is going, not what it already contains.

> "A kernel is not finished when it computes the right answer once. It is finished when you can tell the compiler exactly which memory it may touch, which values it may assume never change, and how many threads get to look at each byte at the same time — and PyTorch's own autograd engine can still find its way through it."

This is a practitioner's guide to [OpenAI Triton](https://triton-lang.org) — the Python-embedded language for writing GPU kernels that compile down to machine code competitive with hand-tuned CUDA, without leaving Python. Unlike a systems language building its own tensor library from scratch, Triton is deliberately narrow: it owns the *kernel*, not the tensor, the device, or the autograd graph. Those belong to PyTorch. The running project in this book is a set of fused, production-style Triton kernels — element-wise ops, a blocked matmul, reductions, fused layer-norm and softmax, Flash Attention, an MoE grouped-GEMM — each one wired into PyTorch's real `torch.autograd.Function` machinery rather than a hand-rolled graph, and each one pointed, in the closing part, at the same quantitative-finance problems the Mojo book ends on.

The design principles carried through the whole book:

- **Kernel-first, not framework-first.** Triton doesn't give you a tensor class or a computational graph — it gives you a compiler for the innermost loop. This book treats that boundary honestly instead of papering over it.
- **PyTorch does the bookkeeping.** Every custom kernel here plugs into `torch.autograd.Function`, so the chain rule, the graph traversal, and gradient accumulation are the real engine's job, not a toy reimplementation of it.
- **Numbers you can check by hand.** Every worked example traces the arithmetic — block indices, strides, masked lanes — before showing the code that produces it.
- **Financial computing ready.** The closing chapters validate these kernels against bond pricing, credit-spread solving, portfolio duration, and Monte Carlo option pricing — domains with zero tolerance for a silently wrong gradient.

<div class="grid cards" markdown>

- :material-book-open-page-variant:{ .lg .middle } **22 chapters**

    Part 0 through Part 7, from `@triton.jit` to Monte Carlo gradients.

- :material-code-tags:{ .lg .middle } **Real, runnable kernels**

    Every section ships runnable Triton source and its captured output — or says plainly when a real run wasn't possible.

- :material-file-document-multiple:{ .lg .middle } **2 appendices**

    Environment setup for a CUDA GPU, plus a self-check quiz.

- :material-finance:{ .lg .middle } **Real financial models**

    Bond pricing and Monte Carlo option pricing, GPU-accelerated.

</div>

## How the book is organized

| Part | Focus |
|---|---|
| **Part 0 — Triton Foundations** | Triton's type system and the JIT boundary, kernel parameterization (`constexpr`, configs), strides and pointers, the grid/program/block model, block-level SPMD vectorization |
| **Part 1 — Kernel–Tensor Interface** | The pointer+stride contract with `torch.Tensor`, broadcasting and block pointers, GPU-side construction, mask/structure kernels (causal, diagonal, block-sparse), targeting hardware, shared memory and the kernel cache |
| **Part 2 — Core Kernels** | Element-wise kernels, the blocked matmul kernel, reduction kernels |
| **Part 3 — Wiring Into PyTorch's Autograd** | `torch.autograd.Function` as the real graph node a Triton kernel plugs into |
| **Part 4 — The Backward Kernel** | Writing the backward kernel and `ctx.save_for_backward`, how PyTorch's autograd engine chains custom ops |
| **Part 5 — GPU Acceleration and Performance** | Autotuning (`num_warps`, `num_stages`, `BLOCK_SIZE`), benchmarking correctly with `triton.testing.do_bench` |
| **Part 6 — Neural Network Building Blocks** | Fused LayerNorm/softmax/dropout/matmul+bias+activation, Flash Attention, MoE grouped-GEMM, higher-order derivatives, gradient checking, serialization |
| **Part 7 — Financial Computing Applications** | Bond pricing, credit spread/risk analytics, portfolio duration, Monte Carlo option pricing |

Start with [Getting Started](getting-started.md) to stand up a Triton + PyTorch + CUDA environment, or jump straight into [Part 0: Triton Foundations](part0/01-types-and-the-jit-boundary.md) if your toolchain is already installed.

!!! note "How this book differs from its Mojo sibling"
    Triton has no user-defined structs, no hand-built tensor class, and no computational graph of its own — a kernel receives raw pointers and strides, and the tensor, the device, and the autograd graph all belong to PyTorch. Rather than mechanically porting chapters that assume a from-scratch systems language, several parts of this book are deliberately redesigned around what Triton actually is: a compiler for the innermost loop that plugs into a real, existing autograd engine. Parts 3 and 4 in particular trade "build your own AD framework" for "build production-grade fused kernels that plug into PyTorch's real one" — which is what Triton is used for in practice.
