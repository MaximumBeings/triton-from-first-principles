import torch
import triton
import triton.language as tl
import numpy as np
from scipy.stats import norm


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


def monte_carlo_call_price(S0, K, T, r, sigma, n_sims, seed=0, BLOCK=1024, discount=True):
    torch.manual_seed(seed)
    z = torch.randn(n_sims)
    payoffs = torch.empty(n_sims)
    grid = (triton.cdiv(n_sims, BLOCK),)
    gbm_call_payoff_kernel[grid](z, payoffs, n_sims, S0, K, T, r, sigma, BLOCK=BLOCK)
    # Independent fixed-draw oracle: validate arithmetic before statistical pricing.
    reference = torch.clamp(S0 * torch.exp((r - 0.5 * sigma ** 2) * T
                            + sigma * (T ** 0.5) * z.double()) - K, min=0.0)
    torch.testing.assert_close(payoffs.double(), reference, rtol=3e-5, atol=3e-4)
    mean_payoff = payoffs.mean()
    se = payoffs.std(unbiased=True) / (n_sims ** 0.5)
    discount_factor = np.exp(-r * T) if discount else 1.0
    price = discount_factor * mean_payoff.item()
    price_se = discount_factor * se.item()
    return price, price_se


def section_22_3():
    print("=== Section 22.3: Monte Carlo GBM pricing vs. Black-Scholes closed form ===")
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    print(f"Black-Scholes closed-form price: {bs_price:.4f}")

    n_sims = 200_000
    mc_price, mc_se = monte_carlo_call_price(S0, K, T, r, sigma, n_sims)
    diff_se = abs(mc_price - bs_price) / mc_se
    print(f"Monte Carlo price: {mc_price:.4f} +/- {mc_se:.4f}, gap = {diff_se:.2f} standard errors")
    print("within 3 standard errors?", diff_se < 3)

    print()
    print("=== Section 22.3: forgetting the discount factor ===")
    mc_price_wrong, mc_se_wrong = monte_carlo_call_price(S0, K, T, r, sigma, n_sims, discount=False)
    diff_se_wrong = abs(mc_price_wrong - bs_price) / mc_se_wrong
    print(f"undiscounted 'price': {mc_price_wrong:.4f}, gap = {diff_se_wrong:.1f} standard errors")
    print("within 3 standard errors?", diff_se_wrong < 3)
    print(f"absolute % difference from true price: {100*abs(mc_price_wrong-bs_price)/bs_price:.1f}% "
          f"(looks small; the standard-error gap says otherwise)")


if __name__ == "__main__":
    section_22_3()
