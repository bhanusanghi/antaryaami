from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Search query model"""

    query: str
    timestamp: datetime = Field(default_factory=datetime.now)


class SearchResult(BaseModel):
    """Search result model"""

    title: str
    link: str
    snippet: str
    timestamp: datetime = Field(default_factory=datetime.now)


class RAGState(BaseModel):
    """State for RAG workflow"""

    query: str
    search_results: List[SearchResult] = Field(default_factory=list)
    error: Optional[str] = None

    def add_error(self, error: str):
        """Add error message"""
        self.error = error
