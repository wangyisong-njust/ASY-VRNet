#!/usr/bin/env python3
"""Average compatible PyTorch checkpoints into a model soup."""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict

import torch


def _state_dict_from_checkpoint(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "net"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value, key
    if isinstance(obj, dict):
        return obj, None
    raise TypeError(f"Unsupported checkpoint type: {type(obj)!r}")


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output checkpoint path.")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths to average.")
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        raise SystemExit("Need at least two checkpoints for a soup.")

    loaded = []
    wrapper_key = None
    for path in args.checkpoints:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        obj = torch.load(path, map_location="cpu")
        state, key = _state_dict_from_checkpoint(obj)
        if wrapper_key is None:
            wrapper_key = key
        elif wrapper_key != key:
            raise ValueError(f"Mixed checkpoint formats: first={wrapper_key!r}, {path}={key!r}")
        normalized = OrderedDict((_strip_module_prefix(k), v) for k, v in state.items())
        loaded.append((path, normalized))

    reference = loaded[0][1]
    averaged = OrderedDict()
    skipped = []
    for name, first_tensor in reference.items():
        tensors = []
        compatible = torch.is_tensor(first_tensor)
        for _, state in loaded:
            value = state.get(name)
            if (not torch.is_tensor(value)) or value.shape != first_tensor.shape:
                compatible = False
                break
            tensors.append(value)
        if compatible and torch.is_floating_point(first_tensor):
            stacked = torch.stack([t.float() for t in tensors], dim=0)
            averaged[name] = stacked.mean(dim=0).to(dtype=first_tensor.dtype)
        else:
            averaged[name] = first_tensor
            if compatible:
                skipped.append(name)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if wrapper_key is None:
        torch.save(averaged, args.out)
    else:
        torch.save({wrapper_key: averaged}, args.out)

    print(f"Saved soup: {args.out}")
    print(f"Inputs: {len(loaded)}")
    for path, _ in loaded:
        print(f"  - {path}")
    print(f"Averaged floating tensors: {len(averaged) - len(skipped)}")
    print(f"Copied non-floating/metadata tensors: {len(skipped)}")


if __name__ == "__main__":
    main()
