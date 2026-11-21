import torch
import triton
import triton.language as tl
import numpy as np
from scipy.stats import norm


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


@triton.jit
def black_scholes_call_kernel_wrong_discount(s_ptr, k_ptr, t_ptr, r_ptr, sigma_ptr, out_ptr, n, BLOCK: tl.constexpr):
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
    call = S * n_d1 - K * n_d2   # BUG: forgot exp(-r*T)
    tl.store(out_ptr + offsets, call, mask=mask)


def run_black_scholes(kernel, S, K, T, r, sigma):
    n = S.numel()
    BLOCK = triton.next_power_of_2(n)
    out = torch.empty_like(S)
    kernel[(triton.cdiv(n, BLOCK),)](S, K, T, r, sigma, out, n, BLOCK=BLOCK)
    return out


def section_22_1():
    print("=== Section 22.1: Black-Scholes kernel vs. scipy-based independent oracle ===")
    S = torch.tensor([100.0, 100.0, 100.0, 80.0, 120.0, 100.0])
    K = torch.tensor([100.0, 90.0, 110.0, 100.0, 100.0, 100.0])
    T = torch.tensor([1.0, 0.5, 2.0, 1.0, 1.0, 0.01])
    r = torch.tensor([0.05, 0.05, 0.03, 0.05, 0.05, 0.05])
    sigma = torch.tensor([0.2, 0.3, 0.25, 0.4, 0.15, 0.2])

    out = run_black_scholes(black_scholes_call_kernel, S, K, T, r, sigma)

    S_np, K_np, T_np, r_np, sigma_np = [x.numpy() for x in (S, K, T, r, sigma)]
    d1 = (np.log(S_np / K_np) + (r_np + 0.5 * sigma_np ** 2) * T_np) / (sigma_np * np.sqrt(T_np))
    d2 = d1 - sigma_np * np.sqrt(T_np)
    oracle = S_np * norm.cdf(d1) - K_np * np.exp(-r_np * T_np) * norm.cdf(d2)

    print("allclose to scipy-based oracle?", np.allclose(out.numpy(), oracle, atol=1e-4))

    print()
    print("=== Section 22.1: forgetting the discount factor ===")
    out_wrong = run_black_scholes(black_scholes_call_kernel_wrong_discount, S, K, T, r, sigma)
    rel_diff_pct = 100 * (out_wrong.numpy() - oracle) / oracle
    print("relative diffs (%):", np.round(rel_diff_pct, 2))
    print("smallest |relative diff| (shortest maturity):", np.round(rel_diff_pct[np.argmin(np.abs(rel_diff_pct))], 2))
    print("largest |relative diff| across these contracts:", np.round(rel_diff_pct[np.argmax(np.abs(rel_diff_pct))], 2))


if __name__ == "__main__":
    section_22_1()
