import json
from .chunker import Chunkernizer
from .loader import JsonLoader
from src.exceptions import CliError, FileError
from .llm import QwenLlm, SematicModel
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
from .cache import (
    save_cache_index,
    load_cache_query,
    save_cache_query,
    save_incremental_index,
    apply_incremental_index,
    incremental_state,
    use_incremental,
    laod_index_cache
)
import fire
from typing import List, Dict, Tuple, Any
from inspect import cleandoc
import bm25s
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict


MAX_CHARACTER = 5000
INDEX_FILE = "data/processed/index.json"


class Rag:
    """Coordinate indexing, retrieval, evaluation, and answer generation."""

    def __init__(self) -> None:
        """Initialize the loader and BM25 retriever."""
        self._chunker: Chunkernizer
        self._loader = JsonLoader()
        self._retriver = bm25s.BM25(method="bm25l", k1=1, b=0.5)

    def index(self, max_chunk_size: int = 2000) -> dict:
        """Build or update the document index.

        Args:
            max_chunk_size: Maximum number of characters per chunk.

        Returns:
            Status information about the indexing operation.
        """
        self._numbers_validator(max_chunk_size)

        value = int(max_chunk_size)

        if value <= 0 or value > 2000:
            raise CliError(
                cleandoc(
                    """
                    Chunk size should be '> 0 and <= 2000'
                    """
                )
            )
        self._chunker = Chunkernizer(value)

        if use_incremental(value):
            files = incremental_state(set(self._chunker.paths))

            if files is not None and (
                files.changed or files.deleted
                or files.new
            ):
                master = apply_incremental_index(
                    files,
                    self._loader,
                    self._chunker,
                )
                self._build_hybrid_index(self._chunker, show=True)

                save_incremental_index(
                    set(self._chunker.paths),
                    value,
                )

                save_cache_index(INDEX_FILE, value)

                return {
                    "status": "Incremental index updated!",
                    "chunks": len(master),
                    "directory": self._loader.chunk_dir + "/processed",
                }
            else:
                path = self._loader.chunk_dir + "/processed/" + "index.json"
                cond, model = laod_index_cache(
                    path,
                    max_chunk_size
                )
                if cond:
                    size = self._loader.load("index.json").size
                    return {
                        "status": "Using cached index!",
                        "chunks": size,
                        "directory": self._loader.chunk_dir + "/processed",
                    }

        self._chunker.files_processor()

        master = self._build_hybrid_index(self._chunker, show=True)

        self._loader.save(master, "index.json")

        save_incremental_index(
            set(self._chunker.paths),
            value,
        )

        save_cache_index(INDEX_FILE, value)

        return {
            "status": "Ingestion complete!",
            "chunks": len(master),
            "directory": self._loader.chunk_dir + "/processed",
        }

    def search(
        self,
        query: str,
        k: int
    ) -> list[str]:
        """Search the indexed corpus using hybrid retrieval.

        Args:
            query: Text to search for.
            k: Maximum number of source references to return.

        Returns:
            Formatted paths and character ranges for matching sources.

        Raises:
            FileError: If the corpus has not been indexed.
        """
        self._numbers_validator(k)
        path = Path(self._loader.chunk_dir) / "processed" / "index.json"
        if not path.exists():
            raise FileError(f"You should index first, we didn't find {path}")
        if not query:
            raise CliError(
                "Please provide a usefull message"
            )
        master = self._loader.load("index.json")
        result = self._search_query(query, k, master)
        out_put = []
        for m in result.retrieved_sources:
            msg = m.file_path
            msg += f" [{m.first_character_index}:"
            msg += f" {m.last_character_index}]"
            out_put.append(msg)
        return out_put

    def semantic_search(self, query: str, k: int) -> list[str]:
        """Search the indexed corpus using semantic similarity.

        Args:
            query: Text to search for.
            k: Maximum number of source references to return.

        Returns:
            Formatted paths and character ranges for matching sources.

        Raises:
            FileError: If the corpus or embeddings have not been indexed.
        """
        self._numbers_validator(k)
        path = Path(self._loader.chunk_dir) / "processed" / "index.json"
        if not path.exists():
            raise FileError(
                f"You should index first, we didn't find {path}"
            )
        embeddings_path = Path("data/processed/embeddings.npy")
        if not embeddings_path.exists():
            raise FileError(
                f"You should index first, we didn't find {embeddings_path}"
            )
        embeddings = np.load("data/processed/embeddings.npy")
        master = self._loader.load("index.json")
        sematic_model = SematicModel()
        result = self._semantic_search(
            query,
            k,
            sematic_model,
            master,
            embeddings
        )
        out_put = []
        for m in result.retrieved_sources:
            msg = m.file_path
            msg += f" [{m.first_character_index}:"
            msg += f" {m.last_character_index}]"
            out_put.append(msg)
        return out_put

    def _hybrid_search(
        self,
        query: str,
        k: int,
        model: SematicModel,
        master: np.ndarray[Any, Any],
        embeddings: Any
    ) -> List[MinimalSource]:
        """Combine semantic and BM25 rankings with rank fusion.

        Args:
            query: Text to search for.
            k: Number of final sources to return.
            model: Semantic embedding model.
            master: Indexed source models.
            embeddings: Embeddings for the indexed sources.

        Returns:
            Sources ordered by the fused ranking.
        """
        if not query:
            raise CliError(
                "Please provide a usefull message"
            )
        semantic = self._semantic_search(
            query,
            k,
            model,
            master,
            embeddings
        )
        search = self._search_query(
            query,
            k,
            master
        )
        final = self._reciprocal_rank_fusion(
            semantic.retrieved_sources,
            search.retrieved_sources
        )
        return final

    def search_dataset(
        self,
        dataset_path: str = "",
        k: int = 0,
        save_directory: str = "",
    ) -> None:
        """Search every question in a dataset and save the results.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of sources to retrieve per question.
            save_directory: Directory where results are written.
        """
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
        path_index = Path(self._loader.chunk_dir) / "processed" / "index.json"
        if not path_index.exists():
            raise FileError(f"You should index first, we didn't find {path}")
        embeddings_path = Path("data/processed/embeddings.npy")
        if not embeddings_path.exists():
            raise FileError(f"You should index first, we didn't find {path}")
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
        embeddings = np.load("data/processed/embeddings.npy")
        master = self._loader.load("index.json")
        sematic_model = SematicModel()
        for question in tqdm(
            data_set.rag_questions,
            desc="searching",
        ):
            try:
                retrived = self._hybrid_search(
                    question.question,
                    k,
                    sematic_model,
                    master,
                    embeddings
                )
            except TypeError:
                raise CliError(
                    f"{question.question} This value broke the program"
                    f"We can't retrive it"
                )
            model = MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=retrived[:k]
            )
            search_results.append(model)

        result = StudentSearchResults(
            search_results=search_results,
            k=k,
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
        k: int = 0
    ) -> str:
        """Retrieve context and generate an answer for one question.

        Args:
            query: Question to answer.
            k: Number of sources to retrieve.

        Returns:
            The generated answer and its retrieved sources.
        """
        if not query:
            raise CliError(
                "Please provide a usefull message"
            )
        self._numbers_validator(k)
        path = Path(self._loader.chunk_dir) / "processed" / "index.json"
        if not path.exists():
            raise FileError(
                f"You should index first, we didn't find {path}"
            )
        embeddings_path = Path("data/processed/embeddings.npy")
        if not embeddings_path.exists():
            raise FileError(
                f"You should index first, we didn't find {embeddings_path}"
            )
        embeddings = np.load("data/processed/embeddings.npy")
        master = self._loader.load("index.json")
        sematic_model = SematicModel()
        question = UnansweredQuestion(question=query)
        try:
            search = self._hybrid_search(
                question.question,
                k,
                sematic_model,
                master,
                embeddings
            )
        except TypeError:
            raise CliError(
                f"{question.question} This value broke the program"
                f"We can't retrive it"
            )
        minimal = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=search
        )
        with tqdm(desc="Generating"):
            obj = self._answer_question(minimal, query, QwenLlm())
        return obj.answer

    def answer_dataset(
        self,
        student_search_results_path: str = "",
        save_directory: str = ""
    ) -> None:
        """Generate answers for a previously searched dataset.

        Args:
            student_search_results_path: Path to saved search results.
            save_directory: Directory where answers are written.
        """
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
        search_result: StudentSearchResults = self._loader.load_obj(
            student_search_results_path
        )
        anwers = []
        model = QwenLlm()
        for m in tqdm(search_result.search_results, desc="LLM working"):
            anwers.append(
                self._answer_question(
                    m,
                    m.question,
                    model
                )
            )
        del model
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

    def evaluate(
        self,
        student_search_results_path: str = "",
        dataset_path: str = "",
    ) -> None:
        """Evaluate retrieval recall against a validated dataset.

        Args:
            student_search_results_path: Path to saved search results.
            dataset_path: Path to the ground-truth dataset.
        """
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

    def _answer_question(
        self,
        search: MinimalSearchResults,
        query: str,
        model: QwenLlm
    ) -> MinimalAnswer:
        """Generate an answer from retrieved context.

        Args:
            search: Retrieved sources and question metadata.
            query: Question text supplied to the language model.
            model: Language model used to generate the answer.

        Returns:
            The generated answer with its retrieval metadata.
        """
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

    @staticmethod
    def _sources_overlap(
        expected_source: MinimalSource | MinimalSourceDataset,
        retrieved_source: MinimalSource,
    ) -> bool:
        """Check whether two source ranges overlap in the same file.

        Args:
            expected_source: Ground-truth source range.
            retrieved_source: Retrieved source range.

        Returns:
            ``True`` when the ranges overlap.
        """
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
        """Build a grounded chat prompt from retrieved source text.

        Args:
            question: User question.
            search_results: Retrieved sources used as context.

        Returns:
            System and user chat messages for the language model.
        """
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

    def _get_metrics(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> Tuple[StudentSearchResults, List[CodeBaseResults]]:
        """Load search results and ground-truth data for evaluation.

        Args:
            student_search_results_path: Path to saved search results.
            dataset_path: Path to the ground-truth dataset.

        Returns:
            The validated search results and evaluation records.

        Raises:
            CliError: If either input path does not exist.
        """
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

    def _numbers_validator(self, value: str | int) -> bool:
        """Validate that a command argument is an integer.

        Args:
            value: Value received from a command or API call.

        Returns:
            ``True`` when the value is a valid integer.

        Raises:
            CliError: If the value is not an integer.
        """
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
        """Build and persist the BM25 index.

        Args:
            chunker: Chunker containing the source chunks.

        Returns:
            All chunks included in the index.
        """
        master = [*chunker.md_chunks, *chunker.chunks, *chunker.py_chunks]
        corpus = [s.text for s in master]
        tokens = bm25s.tokenize(corpus, leave=True)
        self._retriver.index(tokens, leave_progress=True)
        self._retriver.save("data/processed/bm25", leave_progress=True)
        return master

    def _build_hybrid_index(
        self,
        chunker: Chunkernizer,
        show: bool = False
    ) -> List[MinimalSource]:
        """Build and persist both BM25 and semantic indexes.

        Args:
            chunker: Chunker containing the source chunks.
            show: Whether to display embedding progress.

        Returns:
            All chunks included in the indexes.
        """
        master = [*chunker.md_chunks, *chunker.chunks, *chunker.py_chunks]
        corpus = [s.text for s in master]
        tokens = bm25s.tokenize(corpus, leave=True)
        self._retriver.index(tokens, leave_progress=True)
        self._retriver.save("data/processed/bm25", leave_progress=True)
        model = SematicModel()
        embeddings = model.encode(corpus, show)
        np.save("data/processed/embeddings.npy", embeddings)
        return master

    def _semantic_search(
        self,
        query: str,
        k: int,
        model: SematicModel,
        master: np.ndarray[Any, Any],
        embeddings: Any
    ) -> MinimalSearchResults:
        """Retrieve sources ranked by embedding similarity.

        Args:
            query: Text to search for.
            k: Number of sources to retrieve.
            model: Semantic embedding model.
            master: Indexed source models.
            embeddings: Embeddings for the indexed sources.

        Returns:
            Search results ordered by descending similarity.
        """
        if not query.strip():
            raise CliError(
                "Query can't be empty"
            )
        if k <= 0 or k > master.size:
            raise CliError(
                f"k: should be '> 0 and <= {master.size}' <index size>"
            )
        embeddings_query = model.encode([query])[0]
        similaries = embeddings @ embeddings_query
        best_indices = np.argsort(similaries)[::-1][:k]
        result = [
            master[i] for i in best_indices
        ]
        question = UnansweredQuestion(
            question=query
        )
        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=result
        )

    def _search_query(
        self,
        query: str,
        k: int,
        master: np.ndarray[Any, Any]
    ) -> MinimalSearchResults:
        """Retrieve sources ranked by BM25, using the query cache when valid.

        Args:
            query: Text to search for.
            k: Number of sources to retrieve.
            master: Indexed source models.

        Returns:
            Search results ordered by BM25 ranking.

        Raises:
            FileError: If the corpus has not been indexed.
        """
        path = Path(self._loader.chunk_dir) / "processed" / "index.json"
        if not path.exists():
            raise FileError(f"You should index first, we didn't find {path}")
        chunk_size = master.size

        cached = load_cache_query(
            file_path=str(path),
            query=query,
            k=k,
            chunk_size=chunk_size,
        )

        if cached is not None:
            return cached
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
            n_threads=3
        )
        sources = []
        for doc_id in mapped_ids[0]:
            minimal: MinimalSource = master[int(doc_id)]
            sources.append(minimal)
        question = UnansweredQuestion(question=query)
        result = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources
        )
        save_cache_query(
            file_path=str(path),
            query=query,
            k=k,
            chunk_size=master.size,
            search_results=result,
        )
        return result

    def _reciprocal_rank_fusion(
        self,
        *list_ranks: List[MinimalSource],
        k: int = 60
    ) -> List[MinimalSource]:
        """Fuse one or more ranked source lists with RRF.

        Args:
            *list_ranks: Ranked source lists to combine.
            k: RRF rank constant that controls score smoothing.

        Returns:
            Unique sources ordered by their fused score.
        """
        rrf_map: dict[tuple[str, int, int], float] = defaultdict(float)
        items_map: dict[tuple[str, int, int], MinimalSource] = {}
        for rank_list in list_ranks:
            for rank, item in enumerate(rank_list, start=1):
                key = (
                    item.file_path,
                    item.first_character_index,
                    item.last_character_index,
                )
                rrf_map[key] += 1 / (rank + k)
                items_map[key] = item

        sorted_keys = sorted(rrf_map, key=lambda x: rrf_map[x], reverse=True)
        return [items_map[key] for key in sorted_keys]


if __name__ == "__main__":
    cli = Rag()
    fire.Fire(cli)
