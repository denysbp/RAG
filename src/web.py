from fastapi import FastAPI, HTTPException
from .rag import Rag
from .exceptions import CliError

app = FastAPI()
rag = Rag()

@app.post("/index/")
def index(max_chunk_size: int) -> dict:
    try:
        return rag.index(max_chunk_size)
    except CliError as e:
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )

@app.post("/answer/")
def answer(query: str, k: int) -> dict:
    try:
        answer_result = rag.answer(query, str(k))
        return {
            "question_id": answer_result.question_id,
            "question": answer_result.question,
            "answer": answer_result.answer
        }
    except CliError as e:
        raise HTTPException(
            status_code=403,
            detail=str(e)
        )
