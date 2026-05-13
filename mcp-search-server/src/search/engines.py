"""Search engine implementations."""

import logging
from abc import ABC, abstractmethod

from duckduckgo_search import DDGS

from src.config import settings

logger = logging.getLogger(__name__)


class SearchEngineBase(ABC):
    """Base class for search engines."""

    @abstractmethod
    async def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """Perform a search query.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return. Defaults to settings.MAX_RESULTS.

        Returns:
            A list of dictionaries with keys: title, url, snippet.
        """
        pass


class DuckDuckGoEngine(SearchEngineBase):
    """DuckDuckGo search engine implementation (free, no API key required)."""

    async def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """Search using DuckDuckGo.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            A list of dictionaries with keys: title, url, snippet.
        """
        limit = max_results or settings.MAX_RESULTS
        results: list[dict] = []

        try:
            async with DDGS() as ddgs:
                for result in ddgs.text(query, max_results=limit):
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                    })
        except Exception as e:
            logger.error("DuckDuckGo search error: %s", str(e))

        return results


class BingEngine(SearchEngineBase):
    """Bing Search API implementation (requires API key)."""

    async def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """Search using Bing Search API.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            A list of dictionaries with keys: title, url, snippet.
        """
        import httpx

        limit = max_results or settings.MAX_RESULTS
        results: list[dict] = []

        if not settings.SEARCH_API_KEY:
            logger.error("Bing Search API key not configured. Set SEARCH_API_KEY environment variable.")
            return results

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers={"Ocp-Apim-Subscription-Key": settings.SEARCH_API_KEY},
                    params={
                        "q": query,
                        "count": limit,
                        "textDecorations": True,
                        "textFormat": "Raw",
                    },
                )
                response.raise_for_status()
                data = response.json()

                for web_page in data.get("webPages", {}).get("value", []):
                    results.append({
                        "title": web_page.get("name", ""),
                        "url": web_page.get("url", ""),
                        "snippet": web_page.get("snippet", ""),
                    })
        except httpx.HTTPError as e:
            logger.error("Bing Search API error: %s", str(e))
        except Exception as e:
            logger.error("Bing Search error: %s", str(e))

        return results


class GoogleEngine(SearchEngineBase):
    """Google Custom Search API implementation (requires API key and search engine ID)."""

    async def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """Search using Google Custom Search API.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            A list of dictionaries with keys: title, url, snippet.
        """
        import httpx

        limit = max_results or settings.MAX_RESULTS
        results: list[dict] = []

        api_key = settings.SEARCH_API_KEY
        search_engine_id = getattr(settings, "GOOGLE_CSE_ID", "")

        if not api_key or not search_engine_id:
            logger.error(
                "Google Custom Search API key or search engine ID not configured. "
                "Set SEARCH_API_KEY and GOOGLE_CSE_ID environment variables."
            )
            return results

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": api_key,
                        "cx": search_engine_id,
                        "q": query,
                        "num": limit,
                    },
                )
                response.raise_for_status()
                data = response.json()

                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    })
        except httpx.HTTPError as e:
            logger.error("Google Custom Search API error: %s", str(e))
        except Exception as e:
            logger.error("Google Custom Search error: %s", str(e))

        return results


def get_search_engine() -> SearchEngineBase:
    """Get the configured search engine instance.

    Returns:
        An instance of the configured search engine.

    Raises:
        ValueError: If the configured search engine is not supported.
    """
    engines: dict[str, type[SearchEngineBase]] = {
        "duckduckgo": DuckDuckGoEngine,
        "bing": BingEngine,
        "google": GoogleEngine,
    }

    engine_class = engines.get(settings.SEARCH_ENGINE)
    if engine_class is None:
        raise ValueError(
            f"Unknown search engine: {settings.SEARCH_ENGINE}. "
            f"Supported engines: {', '.join(engines.keys())}"
        )

    return engine_class()
