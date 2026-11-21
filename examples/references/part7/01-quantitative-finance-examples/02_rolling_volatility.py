import torch
import triton
import triton.language as tl
import numpy as np
import pandas as pd


@triton.jit
def rolling_std_kernel(x_ptr, out_ptr, n, W, BLOCK_W: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.arange(0, BLOCK_W)
    idx = i - W + 1 + offs
    in_window = offs < W
    in_bounds = (idx >= 0) & (idx < n)
    mask = in_window & in_bounds
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    have_full_window = i >= (W - 1)
    mean = tl.sum(x, axis=0) / W
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / (W - 1)
    std = tl.sqrt(var)

    result = tl.where(have_full_window, std, float("nan"))
    tl.store(out_ptr + i, result)


@triton.jit
def rolling_std_kernel_biased(x_ptr, out_ptr, n, W, BLOCK_W: tl.constexpr):
    i = tl.program_id(0)
    offs = tl.arange(0, BLOCK_W)
    idx = i - W + 1 + offs
    in_window = offs < W
    in_bounds = (idx >= 0) & (idx < n)
    mask = in_window & in_bounds
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    have_full_window = i >= (W - 1)
    mean = tl.sum(x, axis=0) / W
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / W   # BUG (for this convention): biased, divide by W
    std = tl.sqrt(var)

    result = tl.where(have_full_window, std, float("nan"))
    tl.store(out_ptr + i, result)


def rolling_std(x, W, kernel=rolling_std_kernel):
    n = x.numel()
    BLOCK_W = triton.next_power_of_2(W)
    out = torch.empty(n)
    kernel[(n,)](x, out, n, W, BLOCK_W=BLOCK_W)
    return out


def section_22_2():
    print("=== Section 22.2: rolling volatility vs. pandas rolling().std() ===")
    torch.manual_seed(0)
    n, W = 12, 5
    returns = torch.randn(n) * 0.02

    out = rolling_std(returns, W)
    oracle = pd.Series(returns.numpy()).rolling(W).std().values

    nan_match = (np.isnan(out.numpy()) == np.isnan(oracle)).all()
    both_valid = ~np.isnan(out.numpy()) & ~np.isnan(oracle)
    print("NaN positions match?", nan_match)
    print("valid positions allclose?", np.allclose(out.numpy()[both_valid], oracle[both_valid], atol=1e-6))

    print()
    print("=== Section 22.2: the opposite-convention trap (biased vs. unbiased) ===")
    out_biased = rolling_std(returns, W, kernel=rolling_std_kernel_biased)
    valid = ~np.isnan(out.numpy())
    rel_diff = 100 * (out_biased.numpy()[valid] - out.numpy()[valid]) / out.numpy()[valid]
    print("relative diff of biased vs. unbiased (%):", np.round(rel_diff, 2))
    expected = 100 * (np.sqrt((W - 1) / W) - 1)
    print(f"expected constant relative diff for W={W}: {expected:.2f}%")


if __name__ == "__main__":
    section_22_2()
