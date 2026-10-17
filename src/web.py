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
    try:
        return rag.index(max_chunk_size)
    except (CliError, FileError, ValidationError)as e:
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )
    except KeyboardInterrupt:
        exit(1)


@app.post("/search/")
def search(query: str, k: int) -> List:
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
    except KeyboardInterrupt:
        exit(1)


@app.post("/answer/")
def answer(query: str, k: int) -> dict:
    try:
        answer_result = rag.answer(query, k)
        return {
            "question_id": answer_result.question_id,
            "question": answer_result.question,
            "answer": answer_result.answer
        }
    except (CliError, ValidationError, FileError) as e:
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )
    except KeyboardInterrupt:
        exit(1)
