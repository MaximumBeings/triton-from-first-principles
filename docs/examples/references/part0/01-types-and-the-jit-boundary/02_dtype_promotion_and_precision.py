import torch
import triton
import triton.language as tl


@triton.jit
def mixed_dtype_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)   # float32
    y = tl.load(y_ptr + offsets, mask=mask)   # float16
    z = x + y
    tl.static_print("dtype of x:", x.dtype, " dtype of y:", y.dtype, " dtype of z=x+y:", z.dtype)
    tl.store(out_ptr + offsets, z, mask=mask)


@triton.jit
def sum_fp32_accumulator_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)   # fp16
    acc = tl.sum(x.to(tl.float32), axis=0)                # accumulate in fp32
    tl.store(out_ptr, acc)


def demo_implicit_promotion():
    print("=== Implicit dtype promotion: fp32 + fp16 ===")
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float16)
    out = torch.empty(3, dtype=torch.float32)
    mixed_dtype_kernel[(1,)](x, y, out, 3, BLOCK=4)
    print("out:", out.tolist())


def demo_accumulation_dtype_matters():
    print("\n=== Accumulation dtype: naive fp16 vs. fp32-accumulated ===")
    N, VAL = 2048, 0.0001
    big = torch.full((N,), VAL, dtype=torch.float16)

    out_fp32_acc = torch.empty(1, dtype=torch.float32)
    sum_fp32_accumulator_kernel[(1,)](big, out_fp32_acc, N, BLOCK=N)

    acc = torch.tensor(0.0, dtype=torch.float16)
    val = torch.tensor(VAL, dtype=torch.float16)
    for _ in range(N):
        acc = acc + val

    print(f"Summing {N} copies of {VAL} (each stored as fp16):")
    print("  naive sequential fp16 accumulation :", acc.item())
    print("  Triton kernel, fp32 accumulator    :", out_fp32_acc.item())
    print("  exact answer                       :", N * VAL)


if __name__ == "__main__":
    demo_implicit_promotion()
    demo_accumulation_dtype_matters()
