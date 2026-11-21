import torch
import triton
import triton.language as tl


@triton.jit
def matmul_tile_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak
    b_ptrs = b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn
    a = tl.load(a_ptrs, mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
    b = tl.load(b_ptrs, mask=(ks[:, None] < K) & (cols[None, :] < N), other=0.0)
    acc = tl.dot(a, b, input_precision="ieee")
    c_ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


def demo_shared_memory_via_dot():
    print("=== Section 11.3: tl.dot, with zero visibility into real shared-memory staging ===")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    C = torch.empty(2, 2)
    matmul_tile_kernel[(1,)](A, B, C, 2, 2, 2,
                              A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                              BLOCK_M=2, BLOCK_N=2, BLOCK_K=2)
    print("tl.dot result:\n", C)
    print("matches A @ B?", torch.equal(C, A @ B))


if __name__ == "__main__":
    demo_shared_memory_via_dot()
