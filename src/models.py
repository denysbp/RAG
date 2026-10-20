from pydantic import BaseModel, Field
import uuid
from enum import Enum
from typing import List


class MinimalSource(BaseModel):
    """Represent one indexed source fragment and its character bounds."""
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str


class MinimalSourceDataset(BaseModel):
    """Represent source location metadata from an evaluation dataset."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question that has not yet received an answer."""

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question together with expected sources and an answer."""

    sources: List[MinimalSource | MinimalSourceDataset]
    answer: str


class RagDataset(BaseModel):
    """Represent a collection of answered or unanswered RAG questions."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class CodeBaseResults(AnsweredQuestion):
    """Represent an evaluated code-base question and its validity."""

    difficulty: str
    is_valid: bool


class MinimalSearchResults(BaseModel):
    """Represent retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent search results together with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represent search results for a complete dataset search."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent dataset search results after answer generation."""

    search_results: List[MinimalAnswer]
    k: int


class IndexCache(BaseModel):
    """Represent metadata used to validate an index cache."""

    max_chunk_size: int
    last_acess: float
    status: str


class QueryCache(BaseModel):
    """Represent one cached query and its search result."""

    indexed: int
    query: str
    k: int
    last_acess: float
    search_results: MinimalSearchResults


class IncrementalIndex(BaseModel):
    """Represent cached modification metadata for one source file."""

    file_path: str
    last_acess: float


class FilesState(BaseModel):
    """Represent files grouped by their incremental-index state."""

    unchanged: List[str] = Field(default_factory=list)
    changed: List[str] = Field(default_factory=list)
    new: List[str] = Field(default_factory=list)
    deleted: List[str] = Field(default_factory=list)
    size: int


class FileState(Enum):
    """Enumerate possible states of a source file."""

    UNCHANGED = "unchanged"
    CHANGED = "changed"
    NEW = "new"
    DELETED = "deleted"
