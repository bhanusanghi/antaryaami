from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Metadata(BaseModel):
    """Metadata for the prediction question"""

    created_at: datetime = Field(default_factory=datetime.now)
    demographics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    source: Optional[str] = None


class SearchQuery(BaseModel):
    """Search query generated by LLM"""

    query: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Search result from various tools"""

    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """Final prediction response"""

    yes_probability: float = Field(ge=0, le=1)
    no_probability: float = Field(ge=0, le=1)
    confidence_score: float = Field(ge=0, le=1)
    reasoning: Optional[str] = None


class RAGState(BaseModel):
    """State maintained throughout the RAG workflow"""

    original_question: str
    metadata: Optional[Metadata] = None
    search_queries: Optional[List[SearchQuery]] = None
    search_results: Optional[List[SearchResult]] = None
    relevant_chunks: Optional[List[SearchResult]] = None
    prediction: Optional[PredictionResponse] = None
    error: Optional[str] = None
