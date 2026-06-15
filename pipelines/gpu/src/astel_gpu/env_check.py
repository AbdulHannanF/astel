"""Print torch/CUDA/GPU visibility for the astel-gpu environment.

Run with::

    uv run python -m astel_gpu.env_check
"""

from __future__ import annotations

import torch


def main() -> None:
    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.version.cuda = {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {available}")
    count = torch.cuda.device_count()
    print(f"torch.cuda.device_count() = {count}")
    for i in range(count):
        print(f"  device {i}: {torch.cuda.get_device_name(i)}")


if __name__ == "__main__":
    main()
