"""Configuration settings for the MCP Search Server."""

from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    """Server configuration settings."""

    # Server settings
    MCP_SERVER_HOST: str = Field(default="0.0.0.0", description="Host address for the MCP server")
    MCP_SERVER_PORT: int = Field(default=3100, description="Port for the MCP server")

    # Search settings
    SEARCH_ENGINE: Literal["duckduckgo", "bing", "google"] = Field(
        default="duckduckgo", description="Search engine to use"
    )
    SEARCH_API_KEY: str = Field(default="", description="API key for search engine (required for Bing/Google)")
    GOOGLE_CSE_ID: str = Field(default="", description="Google Custom Search Engine ID (required for Google)")

    # Browser settings
    BROWSER_TIMEOUT: int = Field(default=30, description="Browser timeout in seconds")
    SCREENSHOT_DIR: str = Field(default="/app/screenshots", description="Directory to save screenshots")

    # Result settings
    MAX_RESULTS: int = Field(default=10, description="Maximum number of search results")

    # Cache settings
    CACHE_ENABLED: bool = Field(default=True, description="Enable caching")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")

    # Redis settings (optional)
    REDIS_HOST: str = Field(default="", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_PASSWORD: str = Field(default="", description="Redis password")

    @property
    def redis_url(self) -> str | None:
        """Get Redis URL if configured."""
        if not self.REDIS_HOST:
            return None
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"


# Global settings instance
settings = Settings()
