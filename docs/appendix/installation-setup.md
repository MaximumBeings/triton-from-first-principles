# Appendix A: Installation & Setup

## Local machine with an NVIDIA GPU

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton
```

Confirm the driver and toolkit are visible first:

```bash
nvidia-smi
```

If that fails, Triton cannot compile or launch anything — fix the driver install before going further.

## Cloud GPU (no local NVIDIA hardware)

Any CUDA-enabled cloud instance works (a single consumer or datacenter GPU is enough for every kernel in this book). A typical setup on a fresh instance:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton jupyter
```

Notebook-based providers (Colab, Kaggle, most managed notebook platforms) generally ship PyTorch and a compatible Triton pre-installed on their GPU runtimes — check `torch.cuda.is_available()` and `triton.__version__` before installing anything.

## Version notes

This book targets a current stable PyTorch + bundled Triton pairing on CUDA 12.x. Triton's kernel language is stable across recent versions for everything used here, but block-pointer APIs (`tl.make_block_ptr`) and autotuning decorators have shifted signatures across major versions in the past — if an example doesn't match your installed version's behavior, check `triton.__version__` first.

## Editable install (contributing to this book's code)

```bash
git clone https://github.com/MaximumBeings/triton-from-first-principles.git
cd triton-from-first-principles
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install triton mkdocs-material
mkdocs serve
```
