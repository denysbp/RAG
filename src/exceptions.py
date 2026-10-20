class CliError(Exception):
    """Represent an invalid command-line operation."""

    def __init__(self, *args: str) -> None:
        """Initialize the command-line error.

        Args:
            *args: Error message arguments.
        """
        super().__init__(*args)


class ModelsError(Exception):
    """Represent an error involving application models."""

    def __init__(self, *args: str) -> None:
        """Initialize the model error.

        Args:
            *args: Error message arguments.
        """
        super().__init__(*args)


class FileError(Exception):
    """Represent an error while reading or writing a file."""

    def __init__(self, *args: str) -> None:
        """Initialize the file error.

        Args:
            *args: Error message arguments.
        """
        super().__init__(self, *args)


class ChunkError(Exception):
    """Represent an error while creating document chunks."""

    def __init__(self, *args: str) -> None:
        """Initialize the chunking error.

        Args:
            *args: Error message arguments.
        """
        super().__init__(self, *args)


class CacheError(Exception):
    """Represent an error involving cached data."""

    def __init__(self, *args: str) -> None:
        """Initialize the cache error.

        Args:
            *args: Error message arguments.
        """
        super().__init__(self, *args)
