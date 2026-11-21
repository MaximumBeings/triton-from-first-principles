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


def run(kernel, a, b, K_logical, BLOCK_M=4, BLOCK_N=4, BLOCK_K=4):
    M = a.shape[0]
    N = b.shape[1]
    c = torch.empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](a, b, c, M, N, K_logical,
                 a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


def demo_correct_awkward_dims():
    print("=== Section 13.2: correct three-way masking, awkward dimensions ===")
    M, K, N = 5, 7, 6
    A = torch.arange(M * K, dtype=torch.float32).reshape(M, K) * 0.1
    B = torch.arange(K * N, dtype=torch.float32).reshape(K, N) * 0.1
    C = run(matmul_kernel, A, B, K)
    ref = A @ B
    print(f"M,K,N = {M},{K},{N} (none divisible by BLOCK=4)")
    print("matches torch matmul?", torch.allclose(C, ref, atol=1e-3))


def demo_missing_k_mask():
    print("\n=== Section 13.2: omitting the K mask, made visible with a sentinel ===")
    M, K_real, N = 4, 5, 4
    SENTINEL = 999.0
    A_padded = torch.full((M, 8), SENTINEL)
    A_padded[:, :K_real] = torch.arange(M * K_real, dtype=torch.float32).reshape(M, K_real) * 0.1
    A = A_padded[:, :K_real]

    B_padded = torch.full((8, N), SENTINEL)
    B_padded[:K_real, :] = torch.arange(K_real * N, dtype=torch.float32).reshape(K_real, N) * 0.1
    B = B_padded[:K_real, :]

    ref = A @ B
    C_correct = run(matmul_kernel, A, B, K_real)
    C_buggy = run(matmul_kernel_no_k_mask, A, B, K_real)

    print("correct kernel matches?", torch.allclose(C_correct, ref, atol=1e-4))
    print("buggy kernel matches?  ", torch.allclose(C_buggy, ref, atol=1e-4))
    print("buggy kernel result (should be dramatically wrong):\n", C_buggy)


if __name__ == "__main__":
    demo_correct_awkward_dims()
    demo_missing_k_mask()
