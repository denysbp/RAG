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
    RagDataset,
    MinimalAnswer,
    StudentSearchResultsAndAnswer
)
import fire
from typing import List, Dict
from inspect import cleandoc
import bm25s
from pathlib import Path
from tqdm import tqdm
from rich.console import Console
from rich.spinner import Spinner
from rich.text import Text
from rich.live import Live
from nocamel.converters import to_sentence


DEFAULT_CHUNK_SIZE = "2000"
DEFAULT_K = "10"
MAX_CHARACTER = 5000


class Rag:
    def __init__(self) -> None:
        self._chunker: Chunkernizer
        self._loader = JsonLoader()
        self._retriver = bm25s.BM25(method="bm25l", k1=1.5, b=0.5)
        self._aswer_mode: bool = False

    def index(
        self,
        max_chunk_size: str = DEFAULT_CHUNK_SIZE
    ) -> None:
        if self._numbers_validator(max_chunk_size):
            value = int(max_chunk_size)
            if value <= 0 or value > 2000:
                raise CliError(
                    cleandoc(
                        """
                        Chunk size should be '> 0 and  <= 2000'
                        """
                    )
                )
            self._chunker = Chunkernizer(value)
            self._chunker.files_processor()
            master = self._build_index(self._chunker)
            self._loader.save(master, "index.json")
            print(f"Ingestion complete! indexed {len(master)} ", end="")
            print(f"chunks under {self._loader.chunk_dir}")

    def search(
        self,
        query: str,
        k: str = DEFAULT_K
    ) -> MinimalSearchResults:
        query = to_sentence(query)
        self._numbers_validator(k)
        v = not self._aswer_mode
        result = self._search_query(query, int(k), show=v)

        return result

    def search_dataset(
        self,
        dataset_path: str,
        k: str,
        save_directory: str,
    ) -> None:
        if not self._numbers_validator(k):
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

        self._loader.save(
            result.model_dump(),
            save_directory + file_name,
            method="dataset"
        )
        print(f"Saved student_search_results to {save_directory}")

    def answer(
        self,
        query: str,
        k: str = DEFAULT_K
    ) -> MinimalAnswer:
        if not query:
            raise CliError("Provide a usefull message")
        self.aswer_mode = True
        self._model: QwenLlm = QwenLlm()
        search = self._search_query(query, int(k))
        console = Console()
        text = Text("Generating...", style="bold red")
        with tqdm(desc="Generating"):
            obj = self._answer_question(search, query)
        return obj

    def _answer_question(
        self,
        search: MinimalSearchResults,
        query: str
    ) -> MinimalAnswer:
        prompt = self._build_prompt(
            query,
            search
        )
        asnwer = self._model.generate(
            prompt,
            enable_thinking=False
        )

        return MinimalAnswer(
            question_id=search.question_id,
            question=search.question,
            retrieved_sources=search.retrieved_sources,
            answer=asnwer
        )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = ""
    ) -> None:
        path = Path(student_search_results_path)
        if not Path(student_search_results_path).exists():
            msg = cleandoc(
                f"""
                The provider path: {path.absolute()}
                don't exists, please pass a valid path
                and try again
                """
            )
            raise CliError(msg)
        if not save_directory or isinstance(save_directory, bool):
            msg = cleandoc(
                f"""
                The provider path: {save_directory}
                is empty, please provide a valid path
                and try again
                """
            )
            raise CliError(msg)
        console = Console()
        text = Text("Generating...", style="bold red")
        self._model: QwenLlm = QwenLlm()
        search_result = self._loader.load_obj(
            student_search_results_path
        )
        anwers = []
        for m in tqdm(search_result.search_results, desc="LLM working"):
            anwers.append(
                self._answer_question(
                    m,
                    m.question
                )
            )
        result = StudentSearchResultsAndAnswer(
            search_results=anwers,
            k=search_result.k
        )
        file_name = Path(student_search_results_path).name

        self._loader.save(
            result.model_dump(),
            Path(save_directory) / file_name,
            method="dataset"
        )
        print(f"Saved student_search_results to {save_directory}")

    def _build_prompt(
        self,
        question: str,
        search_results: MinimalSearchResults,
    ) -> List[Dict]:
        context = ""
        for search in search_results.retrieved_sources:
            context += search.text + "\n\n"
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

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        pass

    def _numbers_validator(self, value: str) -> bool:
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
        self._retriver.index(tokens, leave_progress=True)
        self._retriver.save("data/processed/bm25", leave_progress=True)
        return master

    def _search_query(
        self,
        query: str,
        k: int,
        show: bool = False
    ) -> MinimalSearchResults:
        path = Path(self._loader.chunk_dir) / "processed" / "index.json"
        if not path.exists():
            raise FileError(f"You should index first, we didn't find {path}")
        master = self._loader.load("index.json")
        if k <= 0 or k > master.size:
            raise CliError(
                f"k: should be '> 0 and <= {master.size}' <index size>"
            )
        mapped = self._retriver.load(
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
