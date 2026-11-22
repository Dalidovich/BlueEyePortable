import sys

import huggingface_hub
import numpy
import PIL
import PySide6
import safetensors
import torch
import transformers


def main() -> int:
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print(f"torch {torch.__version__} (CUDA runtime {torch.version.cuda})")
    print(f"transformers {transformers.__version__}, huggingface_hub {huggingface_hub.__version__}")
    print(f"safetensors {safetensors.__version__}, Pillow {PIL.__version__}, numpy {numpy.__version__}")
    print(f"PySide6 {PySide6.__version__}")
    print(f"CUDA architectures in this build: {' '.join(torch.cuda.get_arch_list())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
