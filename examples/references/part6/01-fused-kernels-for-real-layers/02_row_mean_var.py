import torch
import triton
import triton.language as tl


@triton.jit
def row_mean_var_kernel(x_ptr, mean_ptr, var_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    row_ptr = x_ptr + row * stride_m
    x = tl.load(row_ptr + cols, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N

    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, var)


def row_mean_var(x, BLOCK_N=None):
    M, N = x.shape
    if BLOCK_N is None:
        BLOCK_N = triton.next_power_of_2(N)
    mean = torch.empty(M)
    var = torch.empty(M)
    row_mean_var_kernel[(M,)](x, mean, var, M, N, x.stride(0), BLOCK_N=BLOCK_N)
    return mean, var


def section_20_2():
    print("=== Section 20.2: row-wise mean/variance vs. x.mean()/x.var() ===")
    torch.manual_seed(0)
    M, N = 5, 10
    x = torch.randn(M, N)
    mean, var = row_mean_var(x)

    oracle_mean = x.mean(dim=-1)
    oracle_var_biased = x.var(dim=-1, unbiased=False)
    oracle_var_unbiased = x.var(dim=-1, unbiased=True)

    print("mean allclose to x.mean(dim=-1)?", torch.allclose(mean, oracle_mean, atol=1e-5))
    print("var allclose to x.var(unbiased=False)?", torch.allclose(var, oracle_var_biased, atol=1e-5))
    print("var allclose to x.var(unbiased=True) [WRONG comparison]?",
          torch.allclose(var, oracle_var_unbiased, atol=1e-5))


if __name__ == "__main__":
    section_20_2()
