"""Average LoRA adapters parameter by parameter (e.g. epoch 1 and epoch 2 of one run).

Usage: python avg_adapters.py OUT_DIR ADAPTER_DIR [ADAPTER_DIR ...]
"""

import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    out, srcs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    tensors = [load_file(str(p / "adapter_model.safetensors")) for p in srcs]
    keys = tensors[0].keys()
    assert all(t.keys() == keys for t in tensors), "adapters have different parameters"
    avg = {k: (sum(t[k].float() for t in tensors) / len(tensors)).to(tensors[0][k].dtype) for k in keys}
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(srcs[0] / "adapter_config.json", out / "adapter_config.json")
    save_file(avg, str(out / "adapter_model.safetensors"))
    print(f"averaged {len(srcs)} adapters ({len(keys)} tensors) -> {out}")


if __name__ == "__main__":
    main()
