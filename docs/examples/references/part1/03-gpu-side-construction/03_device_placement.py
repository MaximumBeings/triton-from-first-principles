import torch
import triton
import triton.language as tl


@triton.jit
def add_one_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + 1.0, mask=mask)


def demo_no_cuda_here():
    print("=== Section 8.3: this environment cannot even construct a CUDA tensor ===")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    try:
        t = torch.tensor([1.0, 2.0], device="cuda")
        print("created cuda tensor:", t)
    except Exception as e:
        print(f"attempting to create a real CUDA tensor in this environment -> {type(e).__name__}: {e}")


def demo_interpreter_ignores_device():
    print("\n=== Section 8.3: the interpreter runs a CPU tensor with no complaint ===")
    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    print("x_cpu.device:", x_cpu.device)
    out = torch.empty_like(x_cpu)
    add_one_kernel[(1,)](x_cpu, out, 3, BLOCK=4)
    print("interpreter ran a plain CPU tensor with no complaint:", out.tolist())


if __name__ == "__main__":
    demo_no_cuda_here()
    demo_interpreter_ignores_device()
