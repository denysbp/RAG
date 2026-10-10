from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from pathlib import Path
from typing import Set, List
from .models import MinimalSource
from .exceptions import FileError, ChunkError
from inspect import cleandoc
from .loader import JsonLoader
from tqdm import tqdm


VALID_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
}


class Chunkernizer:
    def __init__(self, chunk_size: int):
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_overlap=0,
            chunk_size=chunk_size,
            add_start_index=True,
            strip_whitespace=False
        )
        self.py_spliter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON,
            chunk_overlap=0,
            chunk_size=chunk_size,
            add_start_index=True,
            strip_whitespace=False
        )
        self.md_spliter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_overlap=0,
            chunk_size=chunk_size,
            add_start_index=True,
            strip_whitespace=False
        )
        self.chunks: List[MinimalSource] = []
        self.py_chunks: List[MinimalSource] = []
        self.md_chunks: List[MinimalSource] = []
        self.vllm_path = Path("data/raw/vllm-0.10.1")
        self.paths: Set[str] = set()
        for file_path in self.vllm_path.rglob("*"):
            if file_path.suffix not in VALID_EXTENSIONS:
                continue
            if file_path.is_file():
                self.paths.add(str(file_path))

    def _chunk_file(self, text: str, path: str) -> None:
        documents = self.spliter.create_documents([text])

        for document in documents:
            source = document.page_content
            start = document.metadata["start_index"]
            end = start + len(source)
            model = MinimalSource.model_validate({
                "file_path": path,
                "first_character_index": start,
                "last_character_index": end,
                "tokens": source
            })
            self.chunks.append(model)

    def _chunk_python_file(self, text: str, path: str) -> None:
        documents = self.py_spliter.create_documents([text])

        for document in documents:
            source = document.page_content
            start = document.metadata["start_index"]
            end = start + len(source)
            model = MinimalSource.model_validate({
                "file_path": path,
                "first_character_index": start,
                "last_character_index": end,
                "tokens": source
            })
            self.py_chunks.append(model)

    def _chunk_markdown_file(self, text: str, path: str) -> None:
        documents = self.md_spliter.create_documents([text])

        for document in documents:
            source = document.page_content
            start = document.metadata["start_index"]
            end = start + len(source)
            model = MinimalSource.model_validate({
                "file_path": path,
                "first_character_index": start,
                "last_character_index": end,
                "tokens": source
            })
            self.md_chunks.append(model)

    def _load_file(self, path: str) -> str:
        try:
            with open(
                path,
                mode="r",
                encoding="utf-8",
                newline=""
            ) as file:
                return file.read()
        except (UnicodeDecodeError, UnicodeError):
            raise FileError(
                cleandoc(
                    """
                    It seems like you passed a file that can't
                    be read with utf-8, we are unable to chunk
                    that file, please remove it, and launch the
                    program again.
                    """
                )
            )
        except IOError as e:
            print(e)
            print("We will skip this file")

    def files_processor(self) -> None:
        for path in tqdm(sorted(self.paths), desc="chunking"):
            text = self._load_file(path)
            if not text:
                continue
            if ".py" in path:
                self._chunk_python_file(text, path)
            elif ".txt" in path:
                self._chunk_file(text, path)
            elif ".md" in path:
                self._chunk_markdown_file(text, path)


if __name__ == "__main__":
    try:
        chunker = Chunkernizer(2000)
        chunker.files_processor()
        loader = JsonLoader()
        loader.save_chunks(chunker.md_chunks, "md_chunks.json")
        loader.save_chunks(chunker.py_chunks, "py_chunks.json")
        loader.save_chunks(chunker.chunks, "chunks.json")
        print(loader.load_chunk("md_chunks.json"))
    except FileError as e:
        print(e.args[1])
