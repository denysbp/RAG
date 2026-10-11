import fire
from src.exceptions import CliError
from .chunker import Chunkernizer
from .loader import JsonLoader
from typing import Any, List
from inspect import cleandoc
from .models import MinimalSource, MinimalSearchResults
import bm25s
from uuid import uuid4

DEFAULT_CHUNK_SIZE = "2000"
DEFAULT_K = "10"


class Rag:
    def __init__(self) -> None:
        self.arguments: dict[str, Any] = {}
        self.chunker: Chunkernizer
        self.loader = JsonLoader()
        self.retriver = bm25s.BM25()

    def index(
        self,
        max_chunk_size: str = DEFAULT_CHUNK_SIZE
    ) -> None:
        if self.numbers_validator(max_chunk_size):
            value = int(max_chunk_size)
            if value <= 0 or value > 2000:
                raise CliError(
                    cleandoc(
                        """
                        Chunk size should be '> 0 and  <= 2000'
                        """
                    )
                )
            self.chunker = Chunkernizer(value)
            self.chunker.files_processor()
            master = self._build_index(self.chunker)
            self.loader.save(master, "master.json")
            print(f"Ingestion complete! indexed {len(master)} ", end="")
            print(f"chunks under {self.loader.chunk_dir}")

    def search(
        self,
        query: str,
        k: str = DEFAULT_K
    ) -> None:
        if self.numbers_validator(k):
            self.loader.save(
                [self._search_query(query, int(k))],
                "minimal_search_results.json",
                dir="search"
            )

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: str = DEFAULT_K,
    ) -> None:
        if self.numbers_validator(k):
            pass

    def answer(
        self,
        query: str,
        k: str = DEFAULT_K
    ) -> None:
        if self.numbers_validator(k):
            pass

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        pass

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        pass

    def numbers_validator(self, value: str) -> bool:
        if not isinstance(value, int) or isinstance(
            value,
            bool
        ):
            msg = cleandoc(
                f"""
                Error: Invalid type was passed on the cli.

                Wrong value: {value},
                we expected integer values.
                """
            )
            raise CliError(msg)
        if isinstance(int(value), int):
            return True
        return False

    def _build_index(self, chunker: Chunkernizer) -> List[MinimalSource]:
        master = [*chunker.chunks, *chunker.py_chunks, *chunker.md_chunks]
        corpus = [s.text for s in master]
        tokens = bm25s.tokenize(corpus, leave=True)
        self.retriver.index(tokens, leave_progress=True)
        self.retriver.save("data/processed/bm25", leave_progress=True)
        return master

    def _search_query(self, query: str, k: int) -> MinimalSearchResults:
        master = self.loader.load("master.json")
        if k <= 0 or k > master.size:
            raise CliError(
                f"k: should be '> 0 and <= {master.size}' <index size>"
            )
        mapped = self.retriver.load(
            "data/processed/bm25",
            mmap=True,
            show_progress=True,
        )
        mapped_ids, _ = mapped.retrieve(
            bm25s.tokenize(query),
            k=k,
            show_progress=True,
        )
        sources = []
        for doc_id in mapped_ids[0]:
            obj: MinimalSource = master[int(doc_id)]
            sources.append(obj)
            index = f"{obj.first_character_index}:{obj.last_character_index}"
            print(obj.file_path, f"[{index}]")
        return MinimalSearchResults(
            question_id=str(uuid4()),
            question=query,
            retrieved_sources=sources
        )


if __name__ == "__main__":
    cli = Rag()
    fire.Fire(cli)
    print(cli.arguments)
