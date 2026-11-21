import torch
import triton
import triton.language as tl


@triton.jit
def copy_block_ptr_kernel(x_ptr, out_ptr, M, N, stride_m, stride_n,
                           out_M, out_N,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M, N), strides=(stride_m, stride_n),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tile = tl.load(x_block_ptr, boundary_check=(0, 1))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr, shape=(out_M, out_N), strides=(out_N, 1),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(out_block_ptr, tile, boundary_check=(0, 1))


def demo_block_pointers():
    print("=== Section 7.3: tl.make_block_ptr, told the truth ===")
    real = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out_correct = torch.empty(3, 4)
    copy_block_ptr_kernel[(1,)](real, out_correct, 3, 4, real.stride(0), real.stride(1), 3, 4, BLOCK_M=4, BLOCK_N=4)
    print("correct block_ptr shape/strides -> matches?", torch.equal(out_correct, real))

    print("\n=== the same Chapter 6 lie, told to a block pointer instead ===")
    out_lied = torch.empty(4, 3)
    copy_block_ptr_kernel[(1,)](real, out_lied, 4, 3, 3, 1, 4, 3, BLOCK_M=4, BLOCK_N=4)
    print("LIED shape (4,3) via block_ptr on a real (3,4) tensor, result:\n", out_lied)
    print("(block_ptr still trusts whatever shape/strides you hand it -- same vulnerability as Chapter 6.2)")


if __name__ == "__main__":
    demo_block_pointers()
