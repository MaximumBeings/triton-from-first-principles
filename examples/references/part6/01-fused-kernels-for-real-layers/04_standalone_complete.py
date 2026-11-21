"""
Chapter 20 standalone: fusing a matmul epilogue with bias+GELU, the row-wise
mean/variance reduction LayerNorm needs, and the complete fused LayerNorm
kernel with its eps-placement trap.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Section 20.1: fused Linear + bias + GELU
# ---------------------------------------------------------------------------

@triton.jit
def fused_linear_gelu_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K,
                              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        b_mask = (ks[:, None] < K) & (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")

    bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.0)
    acc = acc + bias[None, :]
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    cdf = 0.5 * (1.0 + tl.math.erf(acc * SQRT_2_INV))
    result = acc * cdf

    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


def fused_linear_gelu(x, weight, bias, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16):
    M, K = x.shape
    N = weight.shape[0]
    w_t = weight.t().contiguous()
    out = torch.empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fused_linear_gelu_kernel[grid](
        x, w_t, bias, out, M, N, K,
        x.stride(0), x.stride(1), w_t.stride(0), w_t.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def section_20_1():
    print("=== Section 20.1: fused Linear+bias+GELU vs. a real torch.nn.Linear module ===")
    torch.manual_seed(0)
    M, K, N = 20, 12, 16
    layer = torch.nn.Linear(K, N)
    x = torch.randn(M, K)
    with torch.no_grad():
        oracle = torch.nn.functional.gelu(layer(x), approximate="none")
    out = fused_linear_gelu(x, layer.weight.detach(), layer.bias.detach())
    print("allclose to nn.Linear + gelu oracle?", torch.allclose(out, oracle, atol=1e-4))


# ---------------------------------------------------------------------------
# Section 20.2: row-wise mean/variance
# ---------------------------------------------------------------------------

@triton.jit
def row_mean_var_kernel(x_ptr, mean_ptr, var_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    row_ptr = x_ptr + row * stride_m
    x = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, var)


def section_20_2():
    print()
    print("=== Section 20.2: row-wise mean/variance vs. biased and unbiased oracles ===")
    torch.manual_seed(0)
    M, N = 5, 10
    x = torch.randn(M, N)
    BLOCK_N = triton.next_power_of_2(N)
    mean = torch.empty(M)
    var = torch.empty(M)
    row_mean_var_kernel[(M,)](x, mean, var, M, N, x.stride(0), BLOCK_N=BLOCK_N)

    print("mean allclose to x.mean(dim=-1)?", torch.allclose(mean, x.mean(dim=-1), atol=1e-5))
    print("var allclose to x.var(unbiased=False)?",
          torch.allclose(var, x.var(dim=-1, unbiased=False), atol=1e-5))
    print("var allclose to x.var(unbiased=True) [WRONG comparison]?",
          torch.allclose(var, x.var(dim=-1, unbiased=True), atol=1e-5))


# ---------------------------------------------------------------------------
# Section 20.3: complete fused LayerNorm, and the eps-placement trap
# ---------------------------------------------------------------------------

@triton.jit
def layernorm_forward_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr, M, N, stride_m, eps,
                              BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    row_ptr = x_ptr + row * stride_m
    x = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * gamma + beta
    out_row_ptr = out_ptr + row * stride_m
    tl.store(out_row_ptr + cols, y, mask=mask)


@triton.jit
def layernorm_forward_kernel_wrong_eps(x_ptr, gamma_ptr, beta_ptr, out_ptr, M, N, stride_m, eps,
                                        BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    row_ptr = x_ptr + row * stride_m
    x = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / (tl.sqrt(var) + eps)  # BUG
    x_norm = diff * inv_std
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * gamma + beta
    out_row_ptr = out_ptr + row * stride_m
    tl.store(out_row_ptr + cols, y, mask=mask)


def _run_layernorm(x, gamma, beta, eps, kernel):
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)
    out = torch.empty_like(x)
    kernel[(M,)](x, gamma, beta, out, M, N, x.stride(0), eps, BLOCK_N=BLOCK_N)
    return out


def section_20_3():
    print()
    print("=== Section 20.3: fused LayerNorm vs. a real torch.nn.LayerNorm module ===")
    torch.manual_seed(0)
    M, N = 6, 17
    ln = torch.nn.LayerNorm(N)
    x = torch.randn(M, N)
    with torch.no_grad():
        oracle = ln(x)
    out = _run_layernorm(x, ln.weight.detach(), ln.bias.detach(), ln.eps, layernorm_forward_kernel)
    print("allclose to nn.LayerNorm?", torch.allclose(out, oracle, atol=1e-4))

    print()
    print("=== Section 20.3: the eps-placement trap ===")
    torch.manual_seed(1)
    x_typical = torch.randn(4, 8)
    ln2 = torch.nn.LayerNorm(8)
    with torch.no_grad():
        oracle_typical = ln2(x_typical)
    correct_typical = _run_layernorm(x_typical, ln2.weight.detach(), ln2.bias.detach(), ln2.eps,
                                      layernorm_forward_kernel)
    wrong_typical = _run_layernorm(x_typical, ln2.weight.detach(), ln2.bias.detach(), ln2.eps,
                                    layernorm_forward_kernel_wrong_eps)
    print("typical row: correct allclose?", torch.allclose(correct_typical, oracle_typical, atol=1e-4),
          " wrong-eps allclose?", torch.allclose(wrong_typical, oracle_typical, atol=1e-4))

    x_flat = torch.full((1, 8), 5.0) + torch.tensor([[1e-6, -1e-6, 2e-6, 0, -2e-6, 1e-6, 0, -1e-6]])
    ln3 = torch.nn.LayerNorm(8)
    with torch.no_grad():
        oracle_flat = ln3(x_flat)
    correct_flat = _run_layernorm(x_flat, ln3.weight.detach(), ln3.bias.detach(), ln3.eps,
                                   layernorm_forward_kernel)
    wrong_flat = _run_layernorm(x_flat, ln3.weight.detach(), ln3.bias.detach(), ln3.eps,
                                 layernorm_forward_kernel_wrong_eps)
    print("near-constant row: correct allclose?", torch.allclose(correct_flat, oracle_flat, atol=1e-4),
          " wrong-eps allclose?", torch.allclose(wrong_flat, oracle_flat, atol=1e-4))


if __name__ == "__main__":
    section_20_1()
    section_20_2()
    section_20_3()
