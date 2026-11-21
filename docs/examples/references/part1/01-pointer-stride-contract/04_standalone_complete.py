import torch
import triton
import triton.language as tl


@triton.jit
def introspect_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.static_print("x_ptr type:", x_ptr)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def shape_probe_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    tl.static_print("x_ptr.shape as seen inside the kernel:", x_ptr.shape)
    tl.store(out_ptr, 0.0)


@triton.jit
def copy_2d_strided_kernel(x_ptr, out_ptr, M, N, stride_row, stride_col,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    ptrs = x_ptr + rows[:, None] * stride_row + cols[None, :] * stride_col
    v = tl.load(ptrs, mask=mask)
    out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
    tl.store(out_ptrs, v, mask=mask)


@triton.jit
def dtype_report_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.static_print("compiled specialization for dtype:", x.dtype)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, f"shape mismatch: {x.shape} vs {y.shape}"
    assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} vs {y.dtype}"
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


def add_unchecked(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


def main():
    print("=== Section 6.1: what actually crosses into the kernel ===")
    x = torch.randn(4)
    out = torch.empty(4)
    print("host-side: x.shape =", x.shape, " x.stride() =", x.stride(), " x.dtype =", x.dtype)
    introspect_kernel[(1,)](x, out, 4, BLOCK=4)
    shape_probe_kernel[(1,)](x, out, BLOCK=4)
    print("(the real tensor's shape is [4] -- compare against what the kernel saw above)")

    print("\n=== Section 6.2: lying about the shape/stride arguments ===")
    real = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print("real tensor (3x4):\n", real)
    out_correct = torch.empty(3, 4)
    copy_2d_strided_kernel[(1,)](real, out_correct, 3, 4, real.stride(0), real.stride(1), BLOCK_M=4, BLOCK_N=4)
    print("correct M,N,strides -> matches?", torch.equal(out_correct, real))
    out_lied = torch.empty(4, 3)
    copy_2d_strided_kernel[(1,)](real, out_lied, 4, 3, 3, 1, BLOCK_M=4, BLOCK_N=4)
    print("LIED shape (4,3) on a real (3,4) tensor, result:\n", out_lied)

    print("\n=== Section 6.3: dtype triggers its own specialization, automatically ===")
    xf32 = torch.randn(4, dtype=torch.float32)
    outf32 = torch.empty(4, dtype=torch.float32)
    dtype_report_kernel[(1,)](xf32, outf32, 4, BLOCK=4)
    xf16 = torch.randn(4, dtype=torch.float16)
    outf16 = torch.empty(4, dtype=torch.float16)
    dtype_report_kernel[(1,)](xf16, outf16, 4, BLOCK=4)
    xi32 = torch.arange(4, dtype=torch.int32)
    outi32 = torch.empty(4, dtype=torch.int32)
    dtype_report_kernel[(1,)](xi32, outi32, 4, BLOCK=4)

    print("\n=== Section 6.4: the wrapper function is the real composition site ===")
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([10.0, 20.0, 30.0, 40.0])
    result = add(a, b)
    print("add(a, b) =", result.tolist())

    c = torch.tensor([1.0, 2.0, 3.0])
    try:
        add(a, c)
    except AssertionError as e:
        print(f"AssertionError: {e}")

    d = torch.tensor([100.0, 200.0])
    r2 = add_unchecked(a, d)
    print("add_unchecked(a, d) where a has 4 elements, d has 2:", r2.tolist())


if __name__ == "__main__":
    main()
