import fire
from src import CliError
from typing import Any


DEFAULT_CHUNK_SIZE = 2000
DEFAULT_K = 10


class Cli:
    def __init__(self):
        self.arguments: dict[str: Any] = {}

    def index(
        self,
        max_chunk_size: int
    ) -> None:
        self

    def search(
        self,
        query: str,
        k: int
    ) -> None:
        ...

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str
    ) -> None:
        ...

    def answer(
        self,
        query: str,
        k: int
    ) -> None:
        ...

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        ...

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        ...

    def numbers_validator(self, *args: str) -> bool:
        if all(arg for arg in args if arg.isdigit()):
            return True

        invalids = [arg for arg in args if not arg.isdigit()]
        msg = f"""
        Error: Invalids types was passed on the cli.

        We can't tell wich one but they have these values:
        {invalids}, we expected integer values.
        """
        raise CliError(msg)
