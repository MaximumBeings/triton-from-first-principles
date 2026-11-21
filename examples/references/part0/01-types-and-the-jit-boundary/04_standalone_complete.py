import torch
import triton
import triton.language as tl


@triton.jit
def relu_toggle_kernel(x_ptr, out_ptr, n, USE_RELU: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    if USE_RELU:
        x = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def mixed_dtype_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.static_print("dtype of x:", x.dtype, " dtype of y:", y.dtype, " dtype of z=x+y:", z.dtype)
    tl.store(out_ptr + offsets, z, mask=mask)


@triton.jit
def sum_fp32_accumulator_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    acc = tl.sum(x.to(tl.float32), axis=0)
    tl.store(out_ptr, acc)


@triton.jit
def pid_branch_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    if pid == 0:
        x = x * 100.0
    else:
        x = x + 1.0
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def correct_block_conditional_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    x = tl.where(mask, x * 2.0, x)
    tl.store(out_ptr + offsets, x, mask=mask)


def main():
    print("=== Section 1.1/1.2: constexpr-selected kernel variants ===")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, -3.0])
    n = x.shape[0]
    out_relu = torch.empty_like(x)
    relu_toggle_kernel[(1,)](x, out_relu, n, USE_RELU=True, BLOCK=8)
    print("USE_RELU=True  ->", out_relu.tolist())
    out_norelu = torch.empty_like(x)
    relu_toggle_kernel[(1,)](x, out_norelu, n, USE_RELU=False, BLOCK=8)
    print("USE_RELU=False ->", out_norelu.tolist())

    print("\n=== Section 1.3: implicit dtype promotion ===")
    xa = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    ya = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float16)
    outa = torch.empty(3, dtype=torch.float32)
    mixed_dtype_kernel[(1,)](xa, ya, outa, 3, BLOCK=4)
    print("out:", outa.tolist())

    print("\n=== Section 1.3: accumulation dtype matters ===")
    N, VAL = 2048, 0.0001
    big = torch.full((N,), VAL, dtype=torch.float16)
    out_fp32_acc = torch.empty(1, dtype=torch.float32)
    sum_fp32_accumulator_kernel[(1,)](big, out_fp32_acc, N, BLOCK=N)
    acc = torch.tensor(0.0, dtype=torch.float16)
    val = torch.tensor(VAL, dtype=torch.float16)
    for _ in range(N):
        acc = acc + val
    print(f"naive fp16 accumulation: {acc.item()}, fp32-accumulated: {out_fp32_acc.item()}, exact: {N * VAL}")

    print("\n=== Section 1.4: block-shaped values ===")
    xb = torch.arange(6, dtype=torch.float32)
    outb = torch.empty_like(xb)
    pid_branch_kernel[(2,)](xb, outb, 6, BLOCK=4)
    print("scalar branch on tl.program_id:", outb.tolist())
    outc = torch.empty_like(xb)
    correct_block_conditional_kernel[(2,)](xb, outc, 6, BLOCK=4)
    print("tl.where, correct per-lane result:", outc.tolist())


if __name__ == "__main__":
    main()
