# Getting Started

This book assumes an NVIDIA GPU with a recent CUDA driver. Triton compiles kernels to PTX/SASS through CUDA (AMD ROCm support exists but is not what this book targets), so a working `nvidia-smi` is the real prerequisite — everything else is a `pip install` away.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton
```

Recent PyTorch wheels for Linux+CUDA already bundle a matching Triton build, so `pip install triton` on top is often a no-op version pin rather than a fresh install — check with `python -c "import triton; print(triton.__version__)"` before assuming you need it separately.

## Verify the toolchain

```python
import torch
import triton
import triton.language as tl

print("CUDA available:", torch.cuda.is_available())
print("Triton version:", triton.__version__)

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

x = torch.randn(1024, device="cuda")
y = torch.randn(1024, device="cuda")
out = torch.empty_like(x)
_add_kernel[(4,)](x, y, out, 1024, BLOCK=256)
print("Max error vs torch:", (out - (x + y)).abs().max().item())
```

A working setup prints `CUDA available: True`, a Triton version string, and a max error on the order of `1e-7` or smaller (floating-point rounding, not a bug).

## Building this book locally

```bash
pip install mkdocs-material
mkdocs build --strict
mkdocs serve
```

`mkdocs serve` runs a local preview server; `mkdocs build --strict` is what every chapter in this book is checked against before it's committed.

The rendered book will be published at <https://maximumbeings.github.io/triton-from-first-principles/> once the first deploy goes out.

See [Appendix A](appendix/installation-setup.md) for cloud-GPU-specific setup notes.
