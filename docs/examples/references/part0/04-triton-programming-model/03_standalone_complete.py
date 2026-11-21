import torch
import triton
import triton.language as tl


@triton.jit
def global_id_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offsets, offsets)


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


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


def main():
    print("=== Section 4.1: grid of 4 programs, BLOCK=8 ===")
    out = torch.empty(32, dtype=torch.int32)
    global_id_kernel[(4,)](out, BLOCK=8)
    print("all global ids:", out.tolist())
    print("program 2's slice (expect 16..23):", out[16:24].tolist())

    print("\n=== Section 4.2: ceiling-division grid, 1000 elements, BLOCK=256 ===")
    n, BLOCK = 1000, 256
    grid = (triton.cdiv(n, BLOCK),)
    print("triton.cdiv(1000, 256) =", triton.cdiv(n, BLOCK), "-> grid =", grid)
    a = torch.arange(n, dtype=torch.float32)
    b = torch.arange(n, dtype=torch.float32) * 2
    out2 = torch.empty(n, dtype=torch.float32)
    add_kernel[grid](a, b, out2, n, BLOCK=BLOCK)
    print("correct?", torch.equal(out2, a + b))
    total = grid[0] * BLOCK
    print(f"total thread-slots launched: {grid[0]} programs x {BLOCK} = {total}, "
          f"needed: {n}, idle (masked off): {total - n}")

    print("\n=== Section 4.4: naive matmul, one program per output row ===")
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
    main()
