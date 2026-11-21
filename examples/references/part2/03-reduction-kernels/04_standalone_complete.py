import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_max_kernel(x_ptr, sum_ptr, max_ptr, stride_row, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    row_sum = tl.sum(x, axis=0)
    x_for_max = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x_for_max, axis=0)
    tl.store(sum_ptr + row, row_sum)
    tl.store(max_ptr + row, row_max)


@triton.jit
def naive_softmax_kernel(x_ptr, out_ptr, stride_row, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


@triton.jit
def stable_softmax_kernel(x_ptr, out_ptr, stride_row, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    num = tl.exp(x_shifted)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


@triton.jit
def online_softmax_kernel(x_ptr, out_ptr, stride_row, N,
                           BLOCK_N: tl.constexpr, NUM_CHUNKS: tl.constexpr):
    row = tl.program_id(0)
    m = float('-inf')
    l = 0.0
    cols0 = tl.arange(0, BLOCK_N)
    for c in range(NUM_CHUNKS):
        cols = c * BLOCK_N + cols0
        mask = cols < N
        x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        chunk_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, chunk_max)
        safe_max = tl.where(m_new == float('-inf'), 0.0, m_new)
        alpha = tl.where(m == float('-inf'), 0.0, tl.exp(m - safe_max))
        weights = tl.where(mask, tl.exp(x - safe_max), 0.0)
        l = l * alpha + tl.sum(weights, axis=0)
        m = m_new
    for c in range(NUM_CHUNKS):
        cols = c * BLOCK_N + cols0
        mask = cols < N
        x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        safe_max = tl.where(m == float('-inf'), 0.0, m)
        safe_l = tl.where(l > 0.0, l, 1.0)
        y = tl.where(l > 0.0, tl.exp(x - safe_max) / safe_l, 0.0)
        tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


def main():
    print("=== Section 14.1: row sum and row max ===")
    M, N = 4, 6
    x = torch.tensor([
        [1.0, 5.0, 2.0, 0.0, 3.0, -1.0],
        [-2.0, -1.0, -3.0, -4.0, 0.0, 1.0],
        [10.0, 20.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    sums = torch.empty(M)
    maxes = torch.empty(M)
    row_sum_max_kernel[(M,)](x, sums, maxes, x.stride(0), N, BLOCK_N=8)
    print("sums match?", torch.equal(sums, x.sum(dim=1)))
    print("maxes match?", torch.equal(maxes, x.max(dim=1).values))

    print("\n=== Section 14.2: naive vs. stable softmax on large values ===")
    x_large = torch.tensor([[100.0, 101.0, 50.0, 0.0]])
    out_naive = torch.empty_like(x_large)
    naive_softmax_kernel[(1,)](x_large, out_naive, x_large.stride(0), 4, BLOCK_N=8)
    out_stable = torch.empty_like(x_large)
    stable_softmax_kernel[(1,)](x_large, out_stable, x_large.stride(0), 4, BLOCK_N=8)
    oracle_large = torch.softmax(x_large, dim=1)
    print("naive has NaN?", torch.isnan(out_naive).any().item())
    print("stable allclose to oracle?", torch.allclose(out_stable, oracle_large, atol=1e-6))

    print("\n=== Section 14.3: online softmax, multi-chunk row ===")
    N2, BLOCK_N2 = 10, 4
    x2 = torch.tensor([[3.0, 1.0, 0.5, -2.0, 4.0, 0.0, -1.0, 2.0, 5.0, -3.0]])
    out2 = torch.empty_like(x2)
    online_softmax_kernel[(1,)](x2, out2, x2.stride(0), N2, BLOCK_N=BLOCK_N2, NUM_CHUNKS=triton.cdiv(N2, BLOCK_N2))
    oracle2 = torch.softmax(x2, dim=1)
    print("allclose?", torch.allclose(out2, oracle2, atol=1e-6))


if __name__ == "__main__":
    main()
