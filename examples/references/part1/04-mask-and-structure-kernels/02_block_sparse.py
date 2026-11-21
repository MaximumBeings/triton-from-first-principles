import torch
import triton
import triton.language as tl


@triton.jit
def block_sparse_double_kernel(x_ptr, out_ptr, block_mask_ptr, N,
                                BLOCK: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    is_active = tl.load(block_mask_ptr + pid)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    if is_active:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, x * 2.0, mask=mask)


def demo_block_sparse():
    print("=== Section 9.2: skipping whole blocks based on a per-block scalar flag ===")
    N = 16
    BLOCK = 4
    NUM_BLOCKS = triton.cdiv(N, BLOCK)
    x = torch.arange(N, dtype=torch.float32)
    out = torch.zeros(N)
    block_mask = torch.tensor([1, 0, 1, 0], dtype=torch.int32)

    block_sparse_double_kernel[(NUM_BLOCKS,)](x, out, block_mask, N, BLOCK=BLOCK, NUM_BLOCKS=NUM_BLOCKS)
    print("x:            ", x.tolist())
    print("block_mask:   ", block_mask.tolist())
    print("result:       ", out.tolist())


if __name__ == "__main__":
    demo_block_sparse()
