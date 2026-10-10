from .rag import Rag
from .exceptions import *
from fire import Fire


if __name__ == "__main__":
    try:
        Fire(Rag)
    except ChunkError as e:
        print(e)
    except CliError as e:
        print(e)
    except KeyboardInterrupt as e:
        print("Program stoped")