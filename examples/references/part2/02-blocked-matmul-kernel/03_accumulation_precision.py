import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_fp32_acc(a_ptr, b_ptr, c_ptr, M, N, K,
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
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


@triton.jit
def matmul_kernel_fp16_acc(a_ptr, b_ptr, c_ptr, M, N, K,
                            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
        b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
        a_mask = (rows[:, None] < M) & (ks[None, :] < K)
        b_mask = (ks[:, None] < K) & (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        partial = tl.dot(a, b, input_precision="ieee").to(tl.float16)
        acc += partial
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(kernel, a, b, dtype_out, BLOCK_M=4, BLOCK_N=4, BLOCK_K=8):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=dtype_out)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](a, b, c, M, N, K,
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


def demo_accumulation_precision():
    print("=== Section 13.3: fp32 vs fp16 accumulation over a deep K-loop ===")
    torch.manual_seed(0)
    M, K, N = 4, 1024, 4
    A = (torch.randn(M, K) * 0.1).to(torch.float16)
    B = (torch.randn(K, N) * 0.1).to(torch.float16)
    exact = A.double() @ B.double()

    out_fp32acc = run(matmul_kernel_fp32_acc, A, B, torch.float16)
    out_fp16acc = run(matmul_kernel_fp16_acc, A, B, torch.float16)

    print(f"K={K}, BLOCK_K=8 -> {triton.cdiv(K, 8)} K-iterations")
    print("fp32acc max abs diff vs exact:", (out_fp32acc.double() - exact).abs().max().item())
    print("fp16acc max abs diff vs exact:", (out_fp16acc.double() - exact).abs().max().item())


if __name__ == "__main__":
    demo_accumulation_precision()
