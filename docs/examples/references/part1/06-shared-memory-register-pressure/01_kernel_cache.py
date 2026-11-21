import triton
import triton.language as tl
import os


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def demo_kernel_cache():
    interpreted = os.environ.get("TRITON_INTERPRET") == "1"
    print(f"=== Section 11.1: type(add_kernel) under TRITON_INTERPRET={'1' if interpreted else '(unset)'} ===")
    print("type:", type(add_kernel))
    if hasattr(add_kernel, "device_caches"):
        print("device_caches before any launch:", add_kernel.device_caches)
    else:
        try:
            add_kernel.cache
        except AttributeError as e:
            print(f"no cache attribute at all -> AttributeError: {e}")


if __name__ == "__main__":
    demo_kernel_cache()
