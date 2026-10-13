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
    StudentSearchResultsAndAnswer,
    CodeBaseResults,
    MinimalSourceDataset
)
import fire
from typing import List, Dict, Tuple
from inspect import cleandoc
import bm25s
from pathlib import Path
from tqdm import tqdm
from nocamel.converters import to_sentence


DEFAULT_CHUNK_SIZE = "0"
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
        self._numbers_validator(k)
        v = not self._aswer_mode
        result = self._search_query(query, int(k), show=v)

        return result

    def search_dataset(
        self,
        dataset_path: str = "",
        k: str = DEFAULT_K,
        save_directory: str = "",
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
        search_results: List[MinimalSearchResults] = []

        for question in tqdm(
            data_set.rag_questions,
            desc="searching",
        ):
            search_result = self._search_query(
                question.question,
                int(k),
                show=False,
            )

            search_result.question_id = question.question_id

            search_results.append(search_result)

        result = StudentSearchResults(
            search_results=search_results,
            k=int(k),
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
        query: str = "",
        k: str = DEFAULT_K
    ) -> MinimalAnswer:
        if not query:
            raise CliError("Provide a usefull message")
        self.aswer_mode = True
        search = self._search_query(query, int(k))
        with tqdm(desc="Generating"):
            obj = self._answer_question(search, query, QwenLlm())
        return obj

    def _answer_question(
        self,
        search: MinimalSearchResults,
        query: str,
        model: QwenLlm
    ) -> MinimalAnswer:
        prompt = self._build_prompt(
            query,
            search
        )
        asnwer = model.generate(
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
        student_search_results_path: str = "",
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
        search_result = self._loader.load_obj(
            student_search_results_path
        )
        anwers = []
        for m in tqdm(search_result.search_results, desc="LLM working"):
            anwers.append(
                self._answer_question(
                    m,
                    m.question,
                    QwenLlm()
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

    @staticmethod
    def _sources_overlap(
        expected_source: MinimalSource | MinimalSourceDataset,
        retrieved_source: MinimalSource,
    ) -> bool:
        if expected_source.file_path != retrieved_source.file_path:
            return False

        return (
            retrieved_source.first_character_index
            < expected_source.last_character_index
            and expected_source.first_character_index
            < retrieved_source.last_character_index
        )

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
        student_search_results_path: str = "",
        dataset_path: str = "",
    ) -> None:
        search_results, dataset = self._get_metrics(
            student_search_results_path,
            dataset_path,
        )

        dataset_by_id = {
            item.question_id: item
            for item in dataset
            if item.is_valid
        }

        total_relevant_sources_retrieved = 0
        total_relevant_sources = 0
        question_recalls: List[float] = []
        evaluated_questions = 0

        for search_result in tqdm(
            search_results.search_results,
            desc="Evaluating Recall@k",
        ):
            ground_truth = dataset_by_id.get(
                search_result.question_id
            )

            if ground_truth is None:
                continue

            relevant_sources = ground_truth.sources

            if not relevant_sources:
                continue

            retrieved_sources = search_result.retrieved_sources[
                :search_results.k
            ]

            relevant_expected_sources = sum(
                any(
                    self._sources_overlap(
                        expected_source,
                        retrieved_source,
                    )
                    for retrieved_source in retrieved_sources
                )
                for expected_source in relevant_sources
            )

            recall_at_k = (
                relevant_expected_sources / len(relevant_sources)
            )

            question_recalls.append(recall_at_k)
            total_relevant_sources_retrieved += relevant_expected_sources
            total_relevant_sources += len(relevant_sources)
            evaluated_questions += 1

        if evaluated_questions == 0:
            raise CliError(
                "No valid questions with relevant sources "
                "were found for evaluation."
            )

        macro_recall = (
            sum(question_recalls) / evaluated_questions
        )

        micro_recall = (
            total_relevant_sources_retrieved
            / total_relevant_sources
        )

        print(f"Evaluation results (Recall@{search_results.k})")
        print(f"Evaluated questions: {evaluated_questions}")
        print(f"Macro Recall@k: {macro_recall:.4f}")
        print(f"Micro Recall@k: {micro_recall:.4f}")
        print(
            "Relevant sources retrieved: "
            f"{total_relevant_sources_retrieved}/"
            f"{total_relevant_sources}"
        )

    def _get_metrics(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> Tuple[StudentSearchResults, List[CodeBaseResults]]:
        path_search = Path(student_search_results_path)
        path_dataset = Path(dataset_path)
        if not path_dataset.exists():
            raise CliError(
                f"The provider path {path_dataset} don't exist"
            )

        if not path_search.exists():
            raise CliError(
                f"The provider path {path_search} don't exist"
            )
        search = self._loader.load_obj(str(path_search))
        dataset = self._loader.load_data_obj(str(path_dataset))
        return search, dataset

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
            bm25s.tokenize(to_sentence(query)),
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
