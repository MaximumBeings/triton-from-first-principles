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


def triton_matmul(a, b, BLOCK_M=4, BLOCK_N=4, BLOCK_K=4):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


def demo_tiled_matmul():
    print("=== Section 13.1: a real multi-tile matmul, requiring a K-loop and a 2D grid ===")
    M, K, N = 8, 8, 8
    A = torch.arange(M * K, dtype=torch.float32).reshape(M, K) * 0.1
    B = torch.arange(K * N, dtype=torch.float32).reshape(K, N) * 0.1
    C = triton_matmul(A, B, BLOCK_M=4, BLOCK_N=4, BLOCK_K=4)
    ref = A @ B
    print("M,K,N =", M, K, N, " grid =", (triton.cdiv(M, 4), triton.cdiv(N, 4)),
          " K-iterations per tile =", triton.cdiv(K, 4))
    print("matches torch matmul?", torch.allclose(C, ref, atol=1e-3))


if __name__ == "__main__":
    demo_tiled_matmul()
