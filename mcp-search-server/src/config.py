"""Configuration settings for the MCP Search Server."""

from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
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
    BROWSER_HEADLESS: bool = Field(
        default=True,
        description="Run Chromium headless. Set false (with a virtual display like Xvfb) "
        "for the strongest anti-bot evasion — headful browsers are far harder to fingerprint.",
    )
    SCREENSHOT_DIR: str = Field(default="/app/mcp-files/screenshots", description="Directory to save screenshots")
    SESSION_IDLE_TIMEOUT: int = Field(default=600, description="Seconds of inactivity before browser session auto-closes")

    # Result settings
    MAX_RESULTS: int = Field(default=10, description="Maximum number of search results")

    # Fetch settings
    FETCH_MAX_LENGTH: int = Field(default=16000, description="Max content length for fetch tool before summarization")
    FETCH_TOKEN_BUDGET: int = Field(default=4000, description="Token budget for fetch tool content")
    CODE_BLOCK_MAX_CHARS: int = Field(default=2000, description="Max characters per code block in fetch tool output")

    # Output store / pagination settings (read_output tool)
    OUTPUT_STORE_TTL: int = Field(default=1800, description="Seconds to retain large tool outputs for paginated reads via read_output")
    OUTPUT_STORE_MAX_ENTRIES: int = Field(default=128, description="Max stored large outputs before LRU eviction")
    OUTPUT_PREVIEW_CHARS: int = Field(default=4000, description="Default inline preview size before an output is paginated")
    READ_OUTPUT_CHUNK_CHARS: int = Field(default=16000, description="Default characters returned per read_output call")

    # File output settings
    FILE_OUTPUT_DIR: str = Field(default="/app/mcp-files", description="Directory for files created by create_file tool")
    FILE_BASE_URL: str = Field(default="https://mcp.jessfan.com", description="Base URL for downloading files from the MCP server")
    MCP_API_KEY: str = Field(default="", description="MCP server API key")

    # Cache settings
    CACHE_ENABLED: bool = Field(default=True, description="Enable caching (not yet implemented)")
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
