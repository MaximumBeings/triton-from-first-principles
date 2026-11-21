import torch
import triton
import triton.language as tl


@triton.jit
def naive_matmul_row_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                             stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    n_mask = n_offsets < N
    k_mask = k_offsets < K

    a_row = tl.load(a_ptr + row * stride_am + k_offsets * stride_ak, mask=k_mask, other=0.0)
    b_tile = tl.load(
        b_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn,
        mask=(k_mask[:, None] & n_mask[None, :]), other=0.0,
    )
    acc = tl.sum(a_row[:, None] * b_tile, axis=0)
    tl.store(c_ptr + row * stride_cm + n_offsets, acc, mask=n_mask)


def demo_naive_matmul():
    print("=== Section 4.4: naive matmul, one program per output row ===")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N)
    naive_matmul_row_kernel[(M,)](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
        BLOCK_N=2, BLOCK_K=2,
    )
    print("C (Triton):\n", C)
    print("C (torch @):\n", A @ B)
    print("match?", torch.equal(C, A @ B))


if __name__ == "__main__":
    demo_naive_matmul()
