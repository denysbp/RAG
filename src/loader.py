import json
from typing import List
from .models import MinimalSource
from pathlib import Path


class JsonLoader:
    def __init__(self) -> None:
        self.chunk_dir: str = "data/processed/chunks"

    def save_chunks(
        self,
        chunks: List[MinimalSource],
        prefix: str
    ) -> None:
        out_path = Path(self.chunk_dir) / Path(prefix)
        Path(out_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )
        obj = [m.model_dump() for m in chunks]
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(obj, file, indent=2)

    def load_chunk(self, prefix: str) -> List[MinimalSource]:
        path = Path(self.chunk_dir) / Path(prefix)
        with open(path, "r") as file:
            return [
                MinimalSource.model_validate(m) for m in json.load(file)
                ]
