import json
from .chunker import Chunkernizer
from .loader import JsonLoader
from src.exceptions import CliError, FileError
from .llm import QwenLlm
from .models import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    UnansweredQuestion,
    RagDataset
)
import fire
from typing import Any, List, Dict
from inspect import cleandoc
import bm25s
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live


DEFAULT_CHUNK_SIZE = "2000"
DEFAULT_K = "10"
MAX_CHARACTER = 5000


class Rag:
    def __init__(self) -> None:
        self.arguments: dict[str, Any] = {}
        self.chunker: Chunkernizer
        self.loader = JsonLoader()
        self.retriver = bm25s.BM25()
        self.model: QwenLlm = QwenLlm()
        self.aswer_mode: bool = False

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
            v = not self.aswer_mode
            result = self._search_query(query, int(k), show=v)
            obj = StudentSearchResults(
                search_results=[result],
                k=int(k)
            )
            self.loader.save(
                obj.model_dump(),
                "search_results.json",
                "output/search_results"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: str,
        save_directory: str,
    ) -> None:
        if not self.numbers_validator(k):
            return
        path = Path(dataset_path)
        if not Path(dataset_path).exists():
            msg = cleandoc(
                f"""
                The provider path: {path.absolute()}
                don't exists, please pass a valid path
                and try again
                """
            )
            raise CliError(msg)
        with open(path, "r", encoding="utf-8") as file:
            questions = json.load(file)["rag_questions"]
            obj = [
                UnansweredQuestion.model_validate(
                    m
                ) for m in tqdm(questions, desc="loading")
            ]
        data_set = RagDataset(
            rag_questions=obj
        )
        sources = []
        for query in tqdm(data_set.rag_questions, desc="searching"):
            sources.append(
                self._search_query(query.question, int(k), show=False)
            )
        result = StudentSearchResults(
            search_results=sources,
            k=int(k)
        )
        file_name = "/" + str(Path(dataset_path).name)

        self.loader.save(
            result.model_dump(),
            save_directory + file_name,
            method="dataset"
        )
        print(f"Saved student_search_results to {save_directory}")

    def answer(
        self,
        query: str,
        k: str = DEFAULT_K
    ) -> None:
        self.aswer_mode = True
        self.search(query, k)
        search_result = self.loader.load_obj(
            "search_results.json",
            "output/search_results"
        )
        prompt = self._build_prompt(
            query,
            search_result
        )
        console = Console()
        with Live(Spinner("dots", text="Generating..."), console=console):
            txt = self.model.generate(
                prompt,
                enable_thinking=False
            )
        print(txt)


    def _build_prompt(
        self,
        question: str,
        search_results: StudentSearchResults,
    ) -> Dict:
        context = ""
        for search in tqdm(search_results.search_results, desc="Prompting"):
            sources = search.retrieved_sources
            for retrive in sources:
                context += retrive.text + "\n\n"
        system_content = """
        You are a question-answering assistant.

        Answer the user's question using the provided context.
        Focus on information relevant to the question.
        Ignore unrelated code, tests, and examples.
        Do not answer unrelated questions.
        If the context does not contain enough information,
        clearly state that.
        """
        user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
        )
        return [
            {"role": "system", "content": cleandoc(system_content)},
            {"role": "user", "content": user_content},
        ]


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
        master = [*chunker.md_chunks, *chunker.chunks, *chunker.py_chunks]
        corpus = [s.text for s in master]
        tokens = bm25s.tokenize(corpus, leave=True)
        self.retriver.index(tokens, leave_progress=True)
        self.retriver.save("data/processed/bm25", leave_progress=True)
        return master

    def _search_query(
        self,
        query: str,
        k: int,
        show: bool = False
    ) -> MinimalSearchResults:
        path = Path(self.loader.chunk_dir) / "processed" / "master.json"
        if not path.exists():
            raise FileError(f"You should index first, we didn't find {path}")
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
            if show:
                index = f"{obj.first_character_index}:"
                index += f"{obj.last_character_index}"
                print(obj.file_path, f"[{index}]")
        question = UnansweredQuestion(question=query)
        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources
        )


if __name__ == "__main__":
    cli = Rag()
    fire.Fire(cli)
    print(cli.arguments)
