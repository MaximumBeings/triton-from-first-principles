import torch
import triton
import triton.language as tl


def demo_hardware_queries():
    print("=== Section 10.1: querying hardware info without a GPU ===")
    try:
        print("triton.runtime.driver.active:", triton.runtime.driver.active)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

    try:
        target = triton.runtime.driver.active.get_current_target()
        print("current target:", target)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

    print("\ntorch.cuda.is_available():", torch.cuda.is_available())
    try:
        props = torch.cuda.get_device_properties(0)
        print(props)
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    demo_hardware_queries()
