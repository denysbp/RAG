from .rag import Rag
from .exceptions import (
    ChunkError,
    CliError,
    FileError
)
from fire import Fire


if __name__ == "__main__":
    try:
        Fire(Rag)
    except ChunkError as e:
        print(e)
    except CliError as e:
        print(e)
    except FileError as e:
        print(e.args[1])
    except IOError as e:
        print(e)
    except KeyboardInterrupt:
        print("Program stoped")
