"""Search result models."""

from datetime import datetime

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result."""

    url: str = Field(description="URL of the search result")
    title: str = Field(description="Title of the search result")
    snippet: str = Field(description="Snippet or description of the search result")
    engine: str = Field(description="Search engine that returned this result")
    timestamp: datetime = Field(description="Timestamp when the result was fetched", default_factory=datetime.utcnow)


class SearchResponse(BaseModel):
    """Response containing search results."""

    results: list[SearchResult] = Field(default_factory=list, description="List of search results")
    total: int = Field(default=0, description="Total number of results")
    query: str = Field(description="The search query")
