import torch
import triton
import triton.language as tl


@triton.jit
def naive_softmax_kernel(x_ptr, out_ptr, stride_row, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


@triton.jit
def stable_softmax_kernel(x_ptr, out_ptr, stride_row, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float('-inf'))
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    num = tl.exp(x_shifted)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


def run(kernel, x, N, BLOCK_N=8):
    out = torch.empty_like(x)
    M = x.shape[0]
    kernel[(M,)](x, out, x.stride(0), N, BLOCK_N=BLOCK_N)
    return out


def demo_stable_softmax():
    print("=== Section 14.2: ordinary values -- naive and stable agree ===")
    x_small = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    out_naive = run(naive_softmax_kernel, x_small, 4)
    out_stable = run(stable_softmax_kernel, x_small, 4)
    oracle = torch.softmax(x_small, dim=1)
    print("naive allclose?", torch.allclose(out_naive, oracle, atol=1e-6))
    print("stable allclose?", torch.allclose(out_stable, oracle, atol=1e-6))

    print("\n=== Section 14.2: large values -- naive overflows to nan ===")
    x_large = torch.tensor([[100.0, 101.0, 50.0, 0.0]])
    out_naive_large = run(naive_softmax_kernel, x_large, 4)
    out_stable_large = run(stable_softmax_kernel, x_large, 4)
    oracle_large = torch.softmax(x_large, dim=1)
    print("naive result:", out_naive_large.tolist())
    print("naive has NaN?", torch.isnan(out_naive_large).any().item())
    print("stable allclose to oracle?", torch.allclose(out_stable_large, oracle_large, atol=1e-6))


if __name__ == "__main__":
    demo_stable_softmax()
