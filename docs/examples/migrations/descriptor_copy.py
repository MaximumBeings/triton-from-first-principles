"""Descriptor migration example: syntax checked; GPU execution pending.
Requires compatible Linux NVIDIA Triton with host TensorDescriptor support.
"""
import os
if os.environ.get('TRITON_INTERPRET') == '1':
    raise SystemExit('Use the compiled GPU path for this descriptor migration test.')
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def descriptor_copy(desc, out_ptr):
    tile = desc.load([0, 0])
    rows = tl.arange(0, 16)
    cols = tl.arange(0, 16)
    tl.store(out_ptr + rows[:, None] * 16 + cols[None, :], tile)

def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA device required; no execution performed.')
    x = torch.arange(256, device='cuda', dtype=torch.float32).reshape(16,16)
    out = torch.empty_like(x)
    desc = TensorDescriptor(x, list(x.shape), list(x.stride()), [16,16])
    descriptor_copy[(1,)](desc, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    print('DESCRIPTOR COPY PASS: all 256 values checked')

if __name__ == '__main__':
    main()
