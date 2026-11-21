import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def demo_warps_and_stages():
    print("=== Section 10.2: num_warps / num_stages under the interpreter ===")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([10.0, 20.0, 30.0, 40.0])

    for nw in [1, 4, 8]:
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, 4, BLOCK=4, num_warps=nw)
        print(f"num_warps={nw} -> {out.tolist()}")

    try:
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, 4, BLOCK=4, num_stages=3)
        print("num_stages=3 accepted ->", out.tolist())
    except Exception as e:
        print(f"num_stages=3 -> {type(e).__name__}: {e}")


if __name__ == "__main__":
    demo_warps_and_stages()
