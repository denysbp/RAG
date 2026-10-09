from langchain_text_splitters import RecursiveCharacterTextSplitter
import ast
from pathlib import Path
import json
from typing import Set, List
from .models import MinimalSource
from .exceptions import FileError
from inspect import cleandoc


INVALID_EXTENSIONS = {
    ".pyc",
    ".png",
    ".ico",
    ".jpg"
}

class Chunkernizer:
    def __init__(self, chunk_size: int):
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_overlap=0,
            chunk_size=chunk_size,
            add_start_index=True
        )
        self.chunks: list[MinimalSource] = []
        self.vllm_path = Path("data/raw/vllm-0.10.1")
        self.paths: Set[str] = set()
        for file_path in self.vllm_path.rglob("*"):
            if file_path.suffix in INVALID_EXTENSIONS:
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
                "last_character_index": end
            })
            self.chunks.append(model)



    def _chunk_python_file(self, text: str, path: str) -> None:
        pass
    
    def _load_file(self, path: str) -> str:
        try:
            with open(path, mode="r") as file:
                return file.read()
        except (UnicodeDecodeError, UnicodeError):
            raise FileError(
                cleandoc("""
                It seems like you passed a file that can't
                be read with utf-8, we are unable to chunk
                that file, please remove it, and launch the
                program again.
                """
            ))

    def files_processor(self) -> List[MinimalSource]:
        for path in sorted(self.paths):
            text = self._load_file(path)
            if not text:
                continue
            if ".py" in path:
                self._chunk_python_file(text, path)
            else:
                self._chunk_file(text, path)


if __name__ == "__main__":
    try:
        chunker = Chunkernizer(2000)
        chunker.files_processor()
        print("First index----0:",chunker.chunks[0].first_character_index)
        print("First index----1:",chunker.chunks[1].first_character_index)
        print("las index---0:",chunker.chunks[0].last_character_index)
        print("las index---1:",chunker.chunks[1].last_character_index)
    except FileError as e:
        print(e.args[1])