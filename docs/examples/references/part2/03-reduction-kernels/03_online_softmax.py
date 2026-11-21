import torch
import triton
import triton.language as tl


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


def run_online(x, N, BLOCK_N=4):
    out = torch.empty_like(x)
    M = x.shape[0]
    NUM_CHUNKS = triton.cdiv(N, BLOCK_N)
    online_softmax_kernel[(M,)](x, out, x.stride(0), N, BLOCK_N=BLOCK_N, NUM_CHUNKS=NUM_CHUNKS)
    return out


def demo_online_softmax():
    print("=== Section 14.3: online softmax, a row wider than one block ===")
    N, BLOCK_N = 10, 4
    x = torch.tensor([[3.0, 1.0, 0.5, -2.0, 4.0, 0.0, -1.0, 2.0, 5.0, -3.0]])
    out = run_online(x, N, BLOCK_N=BLOCK_N)
    oracle = torch.softmax(x, dim=1)
    print(f"N={N}, BLOCK_N={BLOCK_N} -> {triton.cdiv(N, BLOCK_N)} chunks per row")
    print("allclose?", torch.allclose(out, oracle, atol=1e-6))

    print("\n=== Section 14.3: online softmax with large values spanning multiple chunks ===")
    x_large = torch.tensor([[100.0, 50.0, 101.0, 0.0, 99.0, -5.0, 100.5, 20.0, 0.0, 0.0]])
    out_large = run_online(x_large, N, BLOCK_N=BLOCK_N)
    oracle_large = torch.softmax(x_large, dim=1)
    print("allclose?", torch.allclose(out_large, oracle_large, atol=1e-6))


if __name__ == "__main__":
    demo_online_softmax()
