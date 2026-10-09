from langchain_text_splitters import RecursiveCharacterTextSplitter
import ast
from pathlib import Path
import json
from typing import Set

class Chunkernizer:
    def __init__(self, chunk_size: int):
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_overlap=0,
            chunk_size=chunk_size,
            add_start_index=True
        )
        self.vllm_path = Path("data/raw/vllm-0.10.1")
        self.paths: Set[str]
        for file_path in self.vllm_path.glob("*"):
            if file_path.is_file():
                self.paths.add(file_path)
    
    def _chunk_file(self) -> None:
        pass

    def _chunk_python_file(self) -> None:
        pass


if __name__ == "__main__":
    pass