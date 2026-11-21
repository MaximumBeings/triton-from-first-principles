import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
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
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def matmul_kernel_no_k_mask(a_ptr, b_ptr, c_ptr, M, N, K,
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
        a_mask = (rows[:, None] < M)
        b_mask = (cols[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    c_mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


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


def run(kernel, a, b, dtype_out, BLOCK_M=4, BLOCK_N=4, BLOCK_K=4):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), dtype=dtype_out)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](a, b, c, M, N, K,
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


def main():
    print("=== Section 13.1: tiled matmul, 2x2 grid, 2 K-iterations ===")
    M, K, N = 8, 8, 8
    A = torch.arange(M * K, dtype=torch.float32).reshape(M, K) * 0.1
    B = torch.arange(K * N, dtype=torch.float32).reshape(K, N) * 0.1
    C = run(matmul_kernel, A, B, torch.float32)
    print("matches torch?", torch.allclose(C, A @ B, atol=1e-3))

    print("\n=== Section 13.2: sentinel-based K-mask trap ===")
    M2, K_real, N2 = 4, 5, 4
    SENTINEL = 999.0
    A_padded = torch.full((M2, 8), SENTINEL)
    A_padded[:, :K_real] = torch.arange(M2 * K_real, dtype=torch.float32).reshape(M2, K_real) * 0.1
    Ap = A_padded[:, :K_real]
    B_padded = torch.full((8, N2), SENTINEL)
    B_padded[:K_real, :] = torch.arange(K_real * N2, dtype=torch.float32).reshape(K_real, N2) * 0.1
    Bp = B_padded[:K_real, :]
    ref2 = Ap @ Bp
    C_correct = run(matmul_kernel, Ap, Bp, torch.float32)
    C_buggy = run(matmul_kernel_no_k_mask, Ap, Bp, torch.float32)
    print("correct kernel matches?", torch.allclose(C_correct, ref2, atol=1e-4))
    print("buggy kernel matches?  ", torch.allclose(C_buggy, ref2, atol=1e-4))

    print("\n=== Section 13.3: fp32 vs fp16 accumulation ===")
    torch.manual_seed(0)
    M3, K3, N3 = 4, 1024, 4
    A3 = (torch.randn(M3, K3) * 0.1).to(torch.float16)
    B3 = (torch.randn(K3, N3) * 0.1).to(torch.float16)
    exact = A3.double() @ B3.double()
    out_fp16acc = run(matmul_kernel_fp16_acc, A3, B3, torch.float16, BLOCK_K=8)
    print("fp16acc max abs diff vs exact:", (out_fp16acc.double() - exact).abs().max().item())


if __name__ == "__main__":
    main()
