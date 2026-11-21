import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def local_zeros_kernel(out_ptr, BLOCK: tl.constexpr):
    local = tl.zeros((BLOCK,), dtype=tl.float32)
    local = local + 1.0
    tl.store(out_ptr + tl.arange(0, BLOCK), local)


@triton.jit
def fill_kernel(out_ptr, value, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, value, mask=mask)


def triton_full(shape, value, dtype=torch.float32, device="cpu"):
    out = torch.empty(shape, dtype=dtype, device=device)
    n = out.numel()
    BLOCK = triton.next_power_of_2(n)
    fill_kernel[(1,)](out, value, n, BLOCK=BLOCK)
    return out


@triton.jit
def add_one_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + 1.0, mask=mask)


def main():
    print("=== Section 8.1: no allocation inside a kernel ===")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0])
    out = torch.empty_like(x)
    add_kernel[(1,)](x, y, out, 4, BLOCK=4)
    print("out:", out.tolist())

    try:
        add_kernel[(1,)](x, y, None, 4, BLOCK=4)
        print("unexpectedly succeeded")
    except Exception as e:
        print(f"None as output -> {type(e).__name__}: {e}")

    out2 = torch.empty(4)
    local_zeros_kernel[(1,)](out2, BLOCK=4)
    print("tl.zeros, explicitly stored:", out2.tolist())

    print("\n=== Section 8.2: factory-function-style wrapper ===")
    t = triton_full((6,), 7.0)
    print("triton_full((6,), 7.0) ->", t.tolist())
    t2 = torch.full((6,), 7.0)
    print("torch.full((6,), 7.0)  ->", t2.tolist())

    print("\n=== Section 8.3: device placement ===")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    try:
        torch.tensor([1.0, 2.0], device="cuda")
    except Exception as e:
        print(f"CUDA tensor construction -> {type(e).__name__}: {e}")

    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    out3 = torch.empty_like(x_cpu)
    add_one_kernel[(1,)](x_cpu, out3, 3, BLOCK=4)
    print("interpreter ran a CPU tensor with no complaint:", out3.tolist())


if __name__ == "__main__":
    main()
