import hashlib
from pathlib import Path
import json
from .models import (
    IndexCache,
    QueryCache,
    MinimalSearchResults
)
from typing import Tuple, Optional
from pydantic import ValidationError

CACHE_DIR = Path(".cache")
CACHE_INDEX = CACHE_DIR / "index_cache.json"
CACHE_QUERY = CACHE_DIR / "query_cache.json"
BM25_PATH = Path("data/processed/bm25")


def laod_index_cache(
    file_path: str,
    max_chunk_size: int
) -> Tuple[bool, None | IndexCache]:
    path = Path(file_path)
    if not path.exists():
        return False, None
    if not CACHE_INDEX.exists():
        return False, None
    if not BM25_PATH.exists():
        return False, None
    last_acess = path.stat().st_mtime
    new_index = IndexCache(
        max_chunk_size=max_chunk_size,
        last_acess=last_acess,
        status="Using cached index"
    )
    with open(CACHE_INDEX, "r") as file:
        information = json.load(file)
        if not information:
            return False, None
        last_index = IndexCache.model_validate(information)
        return (
            new_index.last_acess == last_index.last_acess
            and new_index.max_chunk_size == last_index.max_chunk_size,
            last_index
        )


def save_cache_index(
    file_path: str,
    max_chunk_size: int,
) -> None:
    path = Path(file_path)
    CACHE_INDEX.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    with open(CACHE_INDEX, "w") as file:
        obj = IndexCache(
            max_chunk_size=max_chunk_size,
            last_acess=path.stat().st_mtime,
            status="Using cached index"
        )
        json.dump(obj.model_dump(), file, indent=2)


def _query_key(query: str, k: int, chunk_size: int) -> str:
    return json.dumps(
        [query, k, chunk_size],
        ensure_ascii=False,
    )


def load_cache_query(
    file_path: str,
    query: str,
    k: int,
    chunk_size: int,
) -> Optional[MinimalSearchResults]:
    path = Path(file_path)

    if not path.is_file() or not CACHE_QUERY.is_file():
        return None

    try:
        current_mtime = path.stat().st_mtime

        with CACHE_QUERY.open("r", encoding="utf-8") as file:
            cache = json.load(file)

        if not isinstance(cache, dict):
            return None

        information = cache.get(
            _query_key(query, k, chunk_size)
        )

        if information is None:
            return None

        cached = QueryCache.model_validate(information)

        if (
            cached.query != query
            or cached.k != k
            or cached.indexed != chunk_size
            or cached.last_acess != current_mtime
        ):
            return None

        return cached.search_results

    except (
        OSError,
        ValueError,
        TypeError,
        ValidationError,
    ):
        return None


def save_cache_query(
    file_path: str,
    query: str,
    k: int,
    chunk_size: int,
    search_results: MinimalSearchResults,
) -> None:
    path = Path(file_path)

    if not path.is_file():
        return

    CACHE_QUERY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with CACHE_QUERY.open("r", encoding="utf-8") as file:
            cache = json.load(file)

        if not isinstance(cache, dict):
            cache = {}

    except (OSError, ValueError):
        cache = {}

    key = _query_key(query, k, chunk_size)

    entry = QueryCache(
        indexed=chunk_size,
        query=query,
        k=k,
        last_acess=path.stat().st_mtime,
        search_results=search_results,
    )

    cache[key] = entry.model_dump(mode="json")

    with CACHE_QUERY.open("w", encoding="utf-8") as file:
        json.dump(
            cache,
            file,
            indent=2,
            ensure_ascii=False,
        )


def file_hash(path: str) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(8192),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


if __name__ == "__main__":
    save_cache_index("data/processed/index.json", 2000)
    print(laod_index_cache("data/processed/index.json", 2000))
