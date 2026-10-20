import json
from typing import Any
from .models import MinimalSource, StudentSearchResults, CodeBaseResults
from pathlib import Path
import numpy as np
from .exceptions import FileError


class JsonLoader:
    """Load and save the project's JSON and NumPy-backed models."""

    def __init__(self) -> None:
        """Initialize the default data directory."""
        self.chunk_dir: str = "data"

    def save(
        self,
        obj: Any,
        prefix: str | Path,
        dir: str = "processed",
        method: str = "normal"
    ) -> None:
        """Serialize an object to a JSON file.

        Args:
            obj: Object or model collection to serialize.
            prefix: Output filename or path.
            dir: Data subdirectory used in normal mode.
            method: ``"normal"`` for a path under the data directory.
        """
        if method == "normal":
            out_path = Path(self.chunk_dir) / Path(dir) / Path(prefix)
        else:
            out_path = Path(prefix)
        out_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        if isinstance(obj, list):
            obj = [m.model_dump() for m in obj]
        try:
            with open(out_path, "w+", encoding="utf-8") as file:
                json.dump(obj, file, indent=2)
        except IOError:
            raise FileError(
                f"{prefix} we don't have permission to write"
            )

    def load(
        self,
        prefix: str,
        dir: str = "processed",
        obj: Any = MinimalSource
    ) -> np.ndarray[Any, Any]:
        """Load a JSON collection into validated model instances.

        Args:
            prefix: JSON filename under the data directory.
            dir: Data subdirectory containing the file.
            obj: Pydantic model used to validate each item.

        Returns:
            An object-dtype NumPy array of validated models.

        Raises:
            FileError: If the JSON file cannot be read.
        """
        try:
            path = Path(self.chunk_dir) / Path(dir) / Path(prefix)
            with open(path, "r") as file:
                return np.array(
                    [obj.model_validate(m) for m in json.load(file)],
                    dtype=object
                )
        except IOError:
            raise FileError(
                f"{prefix} can't be read"
            )

    def load_obj(
        self,
        prefix: str,
    ) -> StudentSearchResults:
        """Load serialized student search results.

        Args:
            prefix: Path to the search-results JSON file.

        Returns:
            Validated student search results.

        Raises:
            FileError: If the JSON file cannot be read.
        """
        try:
            path = Path(prefix)
            with open(path, "r") as file:
                return StudentSearchResults.model_validate(
                    json.load(file)
                )
        except IOError:
            raise FileError(
                f"{prefix} can't be read"
            )

    def load_data_obj(
            self,
            prefix: str,
    ) -> list[CodeBaseResults]:
        """Load and validate an evaluation dataset.

        Args:
            prefix: Path to the dataset JSON file.

        Returns:
            Validated code-base evaluation records.

        Raises:
            FileError: If the JSON file cannot be read.
        """
        try:
            path = Path(prefix)
            with open(path, "r") as file:
                dataset = json.load(file)["rag_questions"]
                return [
                    CodeBaseResults.model_validate(m) for m in dataset
                ]
        except IOError:
            raise FileError(
                f"{prefix} can't be read"
            )
