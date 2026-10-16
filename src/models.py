from pydantic import BaseModel, Field
import uuid
from enum import Enum
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


class MinimalSourceDataset(BaseModel):
    """
    The MinimalSource model represents
    a single source of information
    """
    file_path: str
    first_character_index: int
    last_character_index: int


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
    sources: List[MinimalSource | MinimalSourceDataset]
    answer: str


class RagDataset(BaseModel):
    """
    The RagDataset model
    represents a dataset of RAG questions
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class CodeBaseResults(AnsweredQuestion):
    difficulty: str
    is_valid: bool


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


class IndexCache(BaseModel):
    max_chunk_size: int
    last_acess: float
    status: str


class QueryCache(BaseModel):
    indexed: int
    query: str
    k: int
    last_acess: float
    search_results: MinimalSearchResults


class IncrementalIndex(BaseModel):
    file_path: str
    last_acess: float
    sources: List[MinimalSource]


class FilesState(BaseModel):
    unchanged: List[str] = Field(default_factory=list)
    changed: List[str] = Field(default_factory=list)
    new: List[str] = Field(default_factory=list)
    deleted: List[str] = Field(default_factory=list)
    size: int


class FileState(Enum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    NEW = "new"
    DELETED = "deleted"
