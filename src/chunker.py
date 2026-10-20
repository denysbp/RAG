from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from pathlib import Path
from typing import Set, List, Optional
from .models import MinimalSource
from .exceptions import FileError
from inspect import cleandoc
from tqdm import tqdm
from rich.text import Text
from rich import print as Print

VALID_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
}


class Chunkernizer:
    """Split supported source files into indexed document chunks."""

    def __init__(self, chunk_size: int):
        """Initialize splitters and discover supported source files.

        Args:
            chunk_size: Maximum number of characters in a chunk.
        """
        if chunk_size < 500:
            msg = cleandoc(
                """
                We have a very large data source,
                so issues such as crashes or slowdowns
                may occur due to the very small number
                of characters per chunk.
                """
            )
            text = Text(
                msg,
                justify="right",
                style="bold red"
            )
            Print(text)
        lap = chunk_size // 8
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_overlap=lap,
            chunk_size=chunk_size,
            add_start_index=True,
            strip_whitespace=False
        )
        self.py_spliter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON,
            chunk_overlap=lap,
            chunk_size=chunk_size,
            add_start_index=True,
            strip_whitespace=False
        )
        self.md_spliter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_overlap=lap,
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

    def _chunk_file(self, text: str, path: str) -> List[MinimalSource]:
        """Split a plain-text file into indexed source models.

        Args:
            text: File contents.
            path: Source file path.

        Returns:
            The generated chunks.
        """
        documents = self.spliter.create_documents([text])
        chunks = []
        for document in documents:
            source = document.page_content
            start = document.metadata["start_index"]
            end = start + len(source)
            model = MinimalSource.model_validate({
                "file_path": path,
                "first_character_index": start,
                "last_character_index": end,
                "text": source.lower()
            })
            chunks.append(model)
        return chunks

    def _chunk_python_file(self, text: str, path: str) -> List[MinimalSource]:
        """Split a Python file using Python-aware separators.

        Args:
            text: File contents.
            path: Source file path.

        Returns:
            The generated Python chunks.
        """
        documents = self.py_spliter.create_documents([text])
        py_chunks = []
        for document in documents:
            source = document.page_content
            start = document.metadata["start_index"]
            end = start + len(source)
            model = MinimalSource.model_validate({
                "file_path": path,
                "first_character_index": start,
                "last_character_index": end,
                "text": source.lower()
            })
            py_chunks.append(model)
        return py_chunks

    def _chunk_markdown_file(
        self,
        text: str,
        path: str
    ) -> List[MinimalSource]:
        """Split a Markdown file using Markdown-aware separators.

        Args:
            text: File contents.
            path: Source file path.

        Returns:
            The generated Markdown chunks.
        """
        documents = self.md_spliter.create_documents([text])
        md_chunks = []
        for document in documents:
            source = document.page_content
            start = document.metadata["start_index"]
            end = start + len(source)
            model = MinimalSource.model_validate({
                "file_path": path,
                "first_character_index": start,
                "last_character_index": end,
                "text": source.lower()
            })
            md_chunks.append(model)
        return md_chunks

    def _load_file(self, path: str) -> str:
        """Read a UTF-8 source file.

        Args:
            path: File path to read.

        Returns:
            The file contents, or an empty string when reading fails.

        Raises:
            FileError: If the file cannot be decoded as UTF-8.
        """
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
            return ""

    def files_processor(self) -> None:
        """Process every discovered supported source file."""
        for path in tqdm(sorted(self.paths), desc="chunking"):
            text = self._load_file(path)
            if not text:
                continue
            if Path(path).suffix.lower() == ".md":
                self.md_chunks.extend(
                    self._chunk_markdown_file(text, path)
                )
            elif Path(path).suffix.lower() == ".txt":
                self.chunks.extend(
                    self._chunk_file(text, path)
                )
            elif Path(path).suffix.lower() == ".py":
                self.py_chunks.extend(
                    self._chunk_python_file(text, path)
                )

    def single_processor(self, file: str) -> Optional[List[MinimalSource]]:
        """Process one supported source file.

        Args:
            file: File path to process.

        Returns:
            The generated chunks, or ``None`` for unsupported or empty files.
        """
        path = Path(file)
        text = self._load_file(file)
        if not text:
            return None
        if path.suffix.lower() == ".md":
            return self._chunk_markdown_file(text, file)
        elif path.suffix.lower() == ".txt":
            return self._chunk_file(text, file)
        elif path.suffix.lower() == ".py":
            return self._chunk_python_file(text, file)
        return None
