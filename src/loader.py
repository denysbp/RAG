import json
from typing import Any
from .models import MinimalSource
from pathlib import Path
import numpy as np


class JsonLoader:
    def __init__(self) -> None:
        self.chunk_dir: str = "data/processed/"

    def save(
        self,
        obj: Any,
        prefix: str,
        dir: str = "chunks",
    ) -> None:
        out_path = Path(self.chunk_dir) / Path(dir) / Path(prefix)
        Path(out_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )
        obj = [m.model_dump() for m in obj]
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(obj, file, indent=2)

    def load(self, prefix: str, dir: str = "chunks") -> np.ndarray[Any, Any]:
        path = Path(self.chunk_dir) / Path(dir) / Path(prefix)
        with open(path, "r") as file:
            return np.array(
                [MinimalSource.model_validate(m) for m in json.load(file)],
                dtype=object
            )
