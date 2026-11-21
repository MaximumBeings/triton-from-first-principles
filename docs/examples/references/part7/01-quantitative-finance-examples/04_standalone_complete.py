"""
Chapter 22 standalone: Black-Scholes as an elementwise kernel (and its
discount-factor trap), rolling-window volatility (and its variance-convention
trap), and Monte Carlo option pricing verified statistically against
Black-Scholes (and its own discount-factor trap, now judged in standard
errors rather than a fixed tolerance).
"""
import torch
import triton
import triton.language as tl
import numpy as np
from scipy.stats import norm
import pandas as pd


# ---------------------------------------------------------------------------
# Section 22.1: Black-Scholes
# ---------------------------------------------------------------------------

@triton.jit
def black_scholes_call_kernel(s_ptr, k_ptr, t_ptr, r_ptr, sigma_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    S = tl.load(s_ptr + offsets, mask=mask, other=1.0)
    K = tl.load(k_ptr + offsets, mask=mask, other=1.0)
    T = tl.load(t_ptr + offsets, mask=mask, other=1.0)
    r = tl.load(r_ptr + offsets, mask=mask, other=0.0)
    sigma = tl.load(sigma_ptr + offsets, mask=mask, other=1.0)
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    sqrt_t = tl.sqrt(T)
    d1 = (tl.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    n_d1 = 0.5 * (1.0 + tl.math.erf(d1 * SQRT_2_INV))
    n_d2 = 0.5 * (1.0 + tl.math.erf(d2 * SQRT_2_INV))
    call = S * n_d1 - K * tl.exp(-r * T) * n_d2
    tl.store(out_ptr + offsets, call, mask=mask)


def section_22_1():
    print("=== Section 22.1: Black-Scholes kernel vs. scipy-based independent oracle ===")
    S = torch.tensor([100.0, 100.0, 100.0, 80.0, 120.0, 100.0])
    K = torch.tensor([100.0, 90.0, 110.0, 100.0, 100.0, 100.0])
    T = torch.tensor([1.0, 0.5, 2.0, 1.0, 1.0, 0.01])
    r = torch.tensor([0.05, 0.05, 0.03, 0.05, 0.05, 0.05])
    sigma = torch.tensor([0.2, 0.3, 0.25, 0.4, 0.15, 0.2])
    n = S.numel()
    BLOCK = triton.next_power_of_2(n)
    out = torch.empty_like(S)
    black_scholes_call_kernel[(triton.cdiv(n, BLOCK),)](S, K, T, r, sigma, out, n, BLOCK=BLOCK)

    S_np, K_np, T_np, r_np, sigma_np = [x.numpy() for x in (S, K, T, r, sigma)]
    d1 = (np.log(S_np / K_np) + (r_np + 0.5 * sigma_np ** 2) * T_np) / (sigma_np * np.sqrt(T_np))
    d2 = d1 - sigma_np * np.sqrt(T_np)
    oracle = S_np * norm.cdf(d1) - K_np * np.exp(-r_np * T_np) * norm.cdf(d2)
    print("allclose to scipy-based oracle?", np.allclose(out.numpy(), oracle, atol=1e-4))


# ---------------------------------------------------------------------------
# Section 22.2: rolling volatility
# ---------------------------------------------------------------------------

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


def section_22_2():
    print()
    print("=== Section 22.2: rolling volatility vs. pandas rolling().std() ===")
    torch.manual_seed(0)
    n, W = 12, 5
    returns = torch.randn(n) * 0.02
    BLOCK_W = triton.next_power_of_2(W)
    out = torch.empty(n)
    rolling_std_kernel[(n,)](returns, out, n, W, BLOCK_W=BLOCK_W)

    oracle = pd.Series(returns.numpy()).rolling(W).std().values
    nan_match = (np.isnan(out.numpy()) == np.isnan(oracle)).all()
    both_valid = ~np.isnan(out.numpy()) & ~np.isnan(oracle)
    print("NaN positions match?", nan_match)
    print("valid positions allclose?", np.allclose(out.numpy()[both_valid], oracle[both_valid], atol=1e-6))


# ---------------------------------------------------------------------------
# Section 22.3: Monte Carlo pricing
# ---------------------------------------------------------------------------

@triton.jit
def gbm_call_payoff_kernel(z_ptr, out_ptr, n, S0, K, T, r, sigma, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * tl.sqrt(T) * z
    S_T = S0 * tl.exp(drift + diffusion)
    payoff = tl.maximum(S_T - K, 0.0)
    tl.store(out_ptr + offsets, payoff, mask=mask)


def section_22_3():
    print()
    print("=== Section 22.3: Monte Carlo GBM pricing vs. Black-Scholes closed form ===")
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    n_sims = 200_000
    torch.manual_seed(0)
    z = torch.randn(n_sims)
    payoffs = torch.empty(n_sims)
    BLOCK = 1024
    gbm_call_payoff_kernel[(triton.cdiv(n_sims, BLOCK),)](z, payoffs, n_sims, S0, K, T, r, sigma, BLOCK=BLOCK)
    mean_payoff = payoffs.mean()
    se = payoffs.std(unbiased=True) / (n_sims ** 0.5)
    mc_price = np.exp(-r * T) * mean_payoff.item()
    mc_se = np.exp(-r * T) * se.item()
    diff_se = abs(mc_price - bs_price) / mc_se

    print(f"Black-Scholes price: {bs_price:.4f}   Monte Carlo price: {mc_price:.4f} +/- {mc_se:.4f}")
    print(f"gap in standard errors: {diff_se:.2f}   within 3 SE? {diff_se < 3}")

    mc_price_wrong = mean_payoff.item()  # no discount
    mc_se_wrong = se.item()
    diff_se_wrong = abs(mc_price_wrong - bs_price) / mc_se_wrong
    print(f"undiscounted gap in standard errors: {diff_se_wrong:.1f}   within 3 SE? {diff_se_wrong < 3}")


if __name__ == "__main__":
    section_22_1()
    section_22_2()
    section_22_3()
