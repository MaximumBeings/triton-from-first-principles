import torch
import triton
import triton.language as tl


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
    inv_std = 1.0 / (tl.sqrt(var) + eps)   # BUG: eps outside the sqrt

    x_norm = diff * inv_std
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * gamma + beta

    out_row_ptr = out_ptr + row * stride_m
    tl.store(out_row_ptr + cols, y, mask=mask)


def triton_layernorm(x, gamma, beta, eps=1e-5, kernel=layernorm_forward_kernel):
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)
    out = torch.empty_like(x)
    kernel[(M,)](x, gamma, beta, out, M, N, x.stride(0), eps, BLOCK_N=BLOCK_N)
    return out


def section_20_3():
    print("=== Section 20.3: fused LayerNorm vs. a real torch.nn.LayerNorm module ===")
    torch.manual_seed(0)
    M, N = 6, 17
    ln = torch.nn.LayerNorm(N)
    x = torch.randn(M, N)
    with torch.no_grad():
        oracle = ln(x)
    out = triton_layernorm(x, ln.weight.detach(), ln.bias.detach(), eps=ln.eps)
    print("allclose to nn.LayerNorm?", torch.allclose(out, oracle, atol=1e-4))

    print()
    print("=== Section 20.3: the eps-placement trap ===")
    torch.manual_seed(1)
    x_typical = torch.randn(4, 8)
    ln2 = torch.nn.LayerNorm(8)
    with torch.no_grad():
        oracle_typical = ln2(x_typical)
    correct_typical = triton_layernorm(x_typical, ln2.weight.detach(), ln2.bias.detach(), eps=ln2.eps)
    wrong_typical = triton_layernorm(x_typical, ln2.weight.detach(), ln2.bias.detach(), eps=ln2.eps,
                                      kernel=layernorm_forward_kernel_wrong_eps)
    print("typical row: correct allclose?", torch.allclose(correct_typical, oracle_typical, atol=1e-4),
          " wrong-eps allclose?", torch.allclose(wrong_typical, oracle_typical, atol=1e-4))

    x_flat = torch.full((1, 8), 5.0) + torch.tensor([[1e-6, -1e-6, 2e-6, 0, -2e-6, 1e-6, 0, -1e-6]])
    ln3 = torch.nn.LayerNorm(8)
    with torch.no_grad():
        oracle_flat = ln3(x_flat)
    correct_flat = triton_layernorm(x_flat, ln3.weight.detach(), ln3.bias.detach(), eps=ln3.eps)
    wrong_flat = triton_layernorm(x_flat, ln3.weight.detach(), ln3.bias.detach(), eps=ln3.eps,
                                   kernel=layernorm_forward_kernel_wrong_eps)
    print("near-constant row: correct allclose?", torch.allclose(correct_flat, oracle_flat, atol=1e-4),
          " wrong-eps allclose?", torch.allclose(wrong_flat, oracle_flat, atol=1e-4))


if __name__ == "__main__":
    section_20_3()
