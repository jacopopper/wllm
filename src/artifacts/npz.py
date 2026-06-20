from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np


def save_npz(path: Path, tensors: dict[str, np.ndarray], *, compressed: bool = True) -> None:
    with path.open("wb") as handle:
        if compressed:
            savez = cast(Any, np.savez_compressed)
        else:
            savez = cast(Any, np.savez)
        savez(handle, **tensors)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
