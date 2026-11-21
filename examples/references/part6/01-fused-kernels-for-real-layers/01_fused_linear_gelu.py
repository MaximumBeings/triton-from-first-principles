import torch
import triton
import triton.language as tl


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


@triton.jit
def fused_linear_gelu_kernel_buggy_bias_axis(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K,
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

    bias = tl.load(bias_ptr + rows, mask=rows < M, other=0.0)  # BUG: wrong axis AND wrong bound
    acc = acc + bias[:, None]
    SQRT_2_INV: tl.constexpr = 0.7071067811865476
    cdf = 0.5 * (1.0 + tl.math.erf(acc * SQRT_2_INV))
    result = acc * cdf

    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


def fused_linear_gelu(x, weight, bias, kernel=fused_linear_gelu_kernel, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16):
    M, K = x.shape
    N = weight.shape[0]
    w_t = weight.t().contiguous()
    out = torch.empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](
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

    print()
    print("=== Section 20.1: the bias-axis bug ===")
    out_buggy = fused_linear_gelu(x, layer.weight.detach(), layer.bias.detach(),
                                   kernel=fused_linear_gelu_kernel_buggy_bias_axis)
    print("buggy kernel allclose to oracle?", torch.allclose(out_buggy, oracle, atol=1e-4))


if __name__ == "__main__":
    section_20_1()
