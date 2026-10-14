from pathlib import Path
import json
from .models import IndexCache
from time import time

CACHE_DIR = Path(".cache")
CACHE_INDEX = CACHE_DIR / "index_cache.json"
BM25_PATH = Path("data/processed/bm25")

def laod_index_cache(file_path: str, max_chunk_size: int) -> bool:
    path =  Path(file_path)
    if not path.exists():
        return False
    if not CACHE_INDEX.exists():
        return False
    if not BM25_PATH.exists():
        return
    last_acess = CACHE_INDEX.stat().st_atime
    new_index = IndexCache(
        max_chunk_size=max_chunk_size,
        last_acess=last_acess
    )
    with open(CACHE_INDEX, "r") as file:
        information = json.load(file)
        if not information:
            return False
        last_index = IndexCache.model_validate(information)
        return (
            new_index.last_acess == last_index.last_acess
            and new_index.max_chunk_size == last_index.max_chunk_size
        )


def save_cache_index(max_chunk_size: int) -> None:
    with open(CACHE_INDEX, "w") as file:
        CACHE_INDEX.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        obj = IndexCache(
            max_chunk_size=max_chunk_size,
            last_acess=0.0
        )
        json.dump(obj, file)


if __name__ == "__main"