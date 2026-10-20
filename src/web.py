from fastapi import FastAPI, HTTPException
from sys import exit
from .rag import Rag
from .exceptions import CliError, FileError
from pydantic import ValidationError
from typing import List
app = FastAPI()
rag = Rag()


@app.post("/index/")
def index(max_chunk_size: int) -> dict:
    """Build the document index for the requested chunk size.

    Args:
        max_chunk_size: Maximum number of characters in each chunk.

    Returns:
        Indexing status information.
    """
    try:
        return rag.index(max_chunk_size)
    except (CliError, FileError, ValidationError)as e:
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )
    except BaseException:
        exit(1)


@app.post("/search/")
def search(query: str, k: int) -> List:
    """Search the indexed corpus for a query.

    Args:
        query: Text to search for.
        k: Maximum number of results to return.

    Returns:
        Matching source references.
    """
    try:
        return rag.search(
            query,
            k
        )
    except (FileError, ValidationError) as e:
        raise HTTPException(
            status_code=403,
            detail=str(e.args[1])
        )
    except BaseException:
        exit(1)


@app.post("/answer/")
def answer(query: str, k: int) -> str:
    """Generate an answer grounded in indexed search results.

    Args:
        query: Question to answer.
        k: Number of sources to retrieve.

    Returns:
        The question identifiers, question text, and generated answer.
    """
    try:
        return rag.answer(query, k)

    except (CliError, ValidationError, FileError) as e:
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )
    except BaseException:
        exit(1)
