from .rag import Rag
from .exceptions import (
    ChunkError,
    CliError,
    FileError
)
from fire import Fire
from pydantic import ValidationError


if __name__ == "__main__":
    try:
        Fire(Rag)
    except (
        ChunkError,
        CliError,
        ValidationError,
        FileError
    ) as e:
        print(e)
    except KeyboardInterrupt:
        print("Program stoped")
