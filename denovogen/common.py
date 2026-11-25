import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def checkpoint(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def restore(path, device="cpu"):
    # Only load checkpoints created by this project or supplied by a trusted source.
    return torch.load(path, map_location=device, weights_only=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def set_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2].cpu())
    if state[3] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state[3]])


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def environment(device):
    import platform

    import rdkit

    return dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        rdkit=rdkit.__version__,
        device=str(device),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        threads=torch.get_num_threads(),
    )
