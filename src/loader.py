import json
from typing import Any
from .models import MinimalSource, StudentSearchResults, CodeBaseResults
from pathlib import Path
import numpy as np


class JsonLoader:
    def __init__(self) -> None:
        self.chunk_dir: str = "data"

    def save(
        self,
        obj: Any,
        prefix: str | Path,
        dir: str = "processed",
        method: str = "normal"
    ) -> None:
        if method == "normal":
            out_path = Path(self.chunk_dir) / Path(dir) / Path(prefix)
        else:
            out_path = Path(prefix)
        out_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        if isinstance(obj, list):
            obj = [m.model_dump() for m in obj]
        with open(out_path, "w+", encoding="utf-8") as file:
            json.dump(obj, file, indent=2)

    def load(
        self,
        prefix: str,
        dir: str = "processed",
        obj: Any = MinimalSource
    ) -> np.ndarray[Any, Any]:
        path = Path(self.chunk_dir) / Path(dir) / Path(prefix)
        with open(path, "r") as file:
            return np.array(
                [obj.model_validate(m) for m in json.load(file)],
                dtype=object
            )

    def load_obj(
        self,
        prefix: str,
    ) -> StudentSearchResults:
        path = Path(prefix)
        with open(path, "r") as file:
            return StudentSearchResults.model_validate(json.load(file))

    def load_data_obj(
            self,
            prefix: str,
    ) -> list[CodeBaseResults]:
        path = Path(prefix)
        with open(path, "r") as file:
            dataset = json.load(file)["rag_questions"]
            return [
                CodeBaseResults.model_validate(m) for m in dataset
            ]
