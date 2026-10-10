import fire
from src.exceptions import CliError
from .chunker import Chunkernizer
from .loader import JsonLoader
from typing import Any
from inspect import cleandoc


DEFAULT_CHUNK_SIZE = "2000"
DEFAULT_K = 10


class Rag:
    def __init__(self) -> None:
        self.arguments: dict[str, Any] = {}
        self.chunker: Chunkernizer
        self.loader = JsonLoader()

    def index(
        self,
        max_chunk_size: str = DEFAULT_CHUNK_SIZE
    ) -> None:
        if self.numbers_validator(max_chunk_size):
            if max_chunk_size > 2000:
                raise CliError(
                    cleandoc(
                        """
                        Chunk sizes must be lower than 2000
                        characters, try pass something above it.
                        """
                    )
                )
            self.chunker = Chunkernizer(int(max_chunk_size))
            self.chunker.files_processor()
            self.loader.save_chunks(
                self.chunker.md_chunks, "md_chunks.json"
            )
            self.loader.save_chunks(
                self.chunker.py_chunks, "py_chunks.json"
            )
            self.loader.save_chunks(
                self.chunker.chunks, "chunks.json"
            )

    def search(
        self,
        query: str,
        k: str = DEFAULT_K
    ) -> None:
        if self.numbers_validator(k):
            pass

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: str = DEFAULT_K,
    ) -> None:
        if self.numbers_validator():
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

        if isinstance(value, int):
            return True
        msg = cleandoc(
        f"""
        Error: Invalid type was passed on the cli.

        Wrong value:
        {value}, we expected integer values.
        """
        )
        raise CliError(msg)


if __name__ == "__main__":
    cli = Rag()
    fire.Fire(cli)
    print(cli.arguments)
