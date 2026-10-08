import fire
from src.exceptions import CliError
from typing import Any


DEFAULT_CHUNK_SIZE = 2000
DEFAULT_K = 10


class Cli:
    def __init__(self):
        self.arguments: dict[str: Any] = {}

    def index(
        self,
        max_chunk_size: str = DEFAULT_CHUNK_SIZE
    ) -> None:
        print(type(max_chunk_size))
        if self.numbers_validator(max_chunk_size):
            self.arguments["index"] = int(max_chunk_size)

    def search(
        self,
        query: str,
        k: str
    ) -> None:
        if self.numbers_validator(k):
            values = {}
            values["query"] = query
            values["k"] = k
            self.arguments["search"] = values.copy()
            del values

    def search_dataset(
        self,
        dataset_path: str,
        k: str,
        save_directory: str
    ) -> None:
        if self.numbers_validator():
            values = {}
            values["dataset_path"] = dataset_path
            values["k"] = k
            values["save_directory"] = save_directory
            self.arguments["search_data"] = values.copy()

    def answer(
        self,
        query: str,
        k: str
    ) -> None:
        if self.numbers_validator(k):
            values = {}
            values["query"] = query
            values["k"] = k
            self.arguments["answer"] = values.copy()

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        values = {}
        values[student_search_results_path] = student_search_results_path
        values[save_directory] = save_directory
        self.arguments["answer_dataset"] = values.copy()

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        values = {}
        values[student_search_results_path] = student_search_results_path
        values[dataset_path] = dataset_path
        self.arguments["evaluate"] = values.copy()

    def numbers_validator(self, *args: str) -> bool:
        print(all(arg for arg in args if arg.isdigit()))
        if all(arg for arg in args if arg.isdigit()):
            return True

        invalids = [arg for arg in args if not arg.isdigit()]
        msg = f"""
        Error: Invalids types was passed on the cli.

        We can't tell wich one but they have these values:
        {invalids}, we expected integer values.
        """
        raise CliError(msg)


if __name__ == "__main__":
    cli = Cli()
    fire.Fire(cli)
    print(cli.arguments)
