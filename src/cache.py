from .models import (
    IndexCache,
    QueryCache,
    MinimalSearchResults,
    IncrementalIndex,
    MinimalSource,
    FilesState,
    FileState
)
from .chunker import Chunkernizer
from typing import (
    Tuple,
    Optional,
    List,
    Set,
    Any,
)
from pydantic import ValidationError
from pathlib import Path
import json
from tqdm import tqdm


CACHE_DIR = Path(".cache")
CACHE_INDEX = CACHE_DIR / "index_cache.json"
CACHE_QUERY = CACHE_DIR / "query_cache.json"
INCREMENTAL_INDEX = CACHE_DIR / "incremental_cache.json"
BM25_PATH = Path("data/processed/bm25")
MASTER_PATH = Path("data/processed/index.json")


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

        cached: QueryCache = QueryCache.model_validate(information)

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


def file_hash(path: str) -> float:
    return Path(path).stat().st_mtime


def get_chunks(file_path: str) -> List[MinimalSource]:
    with MASTER_PATH.open("r", encoding="utf-8") as file:
        return [
            MinimalSource.model_validate(m)
            for m in json.load(file)
            if m["file_path"] == file_path
        ]


def save_incremental_index(files: Set[str], size: int) -> None:
    INCREMENTAL_INDEX.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sources: dict[Any, Any] = {}
    for file in tqdm(files, desc="Saving cache"):
        obj = IncrementalIndex(
            file_path=file,
            last_acess=file_hash(file),
            sources=get_chunks(file)
        ).model_dump()
        sources[file] = obj
    sources["paths"] = list(files)
    sources["size"] = size
    with open(INCREMENTAL_INDEX, "w") as file:
        json.dump(sources, file, indent=2)


def get_state(file_path: str, data: IncrementalIndex) -> FileState:
    path = Path(file_path)
    if not path.exists():
        return FileState.DELETED
    last_acess = path.stat().st_mtime
    if last_acess != data.last_acess:
        return FileState.CHANGED
    if last_acess == data.last_acess:
        return FileState.UNCHANGED
    return FileState.NEW


def incremental_state(files: Set[str]) -> Optional[FilesState]:
    states: dict[Any, List] = {}
    states[FileState.UNCHANGED.value] = []
    states[FileState.CHANGED.value] = []
    states[FileState.NEW.value] = []
    states[FileState.DELETED.value] = []
    with INCREMENTAL_INDEX.open("r") as file:
        cache: dict = json.load(file)

        deleted_files = set(cache["paths"]) - files
        states[FileState.DELETED.value].extend(
            deleted_files
        )
        if not cache:
            return None
        for file in tqdm(files, desc="Detecting"):
            obj = cache.get(file)
            if not obj:
                states[FileState.NEW.value].append(file)
                continue
            model = IncrementalIndex.model_validate(obj)
            state = get_state(
                file,
                model
            )
            states[state.value].append(file)
    states["size"] = cache["size"]
    return FilesState.model_validate(states)


def apply_changed(
    file: str,
    chunker: Chunkernizer,
    master: List[MinimalSource]
) -> List[MinimalSource]:
    chunks = chunker.single_processor(file)
    valids = [
        chunk for chunk in master
        if chunk.file_path != file
    ]
    valids.extend(
        chunks
    )
    return valids


def apply_deleted(file: str) -> None:
    ...


def apply_new(file: str) -> None:
    ...


def aplly_incremental_index(files: FilesState) -> None:
    for model in files.changed:
        apply_changed(model)
    for model in files.deleted:
        apply_deleted(model)
    for model in files.new:
        apply_new(model)


if __name__ == "__main__":
    chunker = Chunkernizer(2000)
    save_incremental_index(chunker.paths, 2000)
    # print(len(incremental_state(chunker.paths).deleted))
