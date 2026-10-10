from pydantic import BaseModel, Field
import uuid
from typing import List


class MinimalSource(BaseModel):
    """
    The MinimalSource model represents
    a single source of information
    """
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


"""
The UnansweredQuestion and
AnsweredQuestion models represent an unanswered
question and an answered question:
"""


class UnansweredQuestion(BaseModel):
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    The RagDataset model
    represents a dataset of RAG questions
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


"""
The MinimalSearchResults and MinimalAnswer
models represent the search results
and an answer
"""


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


"""
The StudentSearchResults and StudentSearchResultsAndAnswer
models represent
search results and search results with answers:
"""


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
