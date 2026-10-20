from .rag import Rag
from .exceptions import (
    ChunkError,
    CliError,
    FileError,
    CacheError
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
        FileError,
        CacheError
    ) as e:
        print(e)
    except KeyboardInterrupt:
        print("Program stopped")
    except Exception as e:
        print(f"A Error occurred: {e}")
