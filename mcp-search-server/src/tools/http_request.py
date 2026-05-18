"""Generic HTTP request tool for MCP server."""

import json
import logging
from typing import Any

import httpx
from mcp.server import FastMCP

from src.config import settings

logger = logging.getLogger(__name__)


# Methods that allow request body
_METHODS_WITH_BODY = {"POST", "PUT", "PATCH"}


def http_request_handler(server: FastMCP) -> None:
    """Register the http_request tool."""

    @server.tool()
    async def http_request(
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: str | None = None,
        auth: dict[str, str] | None = None,
        timeout: int | None = None,
        follow_redirects: bool = True,
    ) -> str:
        """Make an HTTP request to any URL.

        Supports GET, POST, PUT, PATCH, DELETE, HEAD, and OPTIONS methods.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS)
            url: The URL to request
            headers: Optional dictionary of HTTP headers
            body: Optional request body (for POST, PUT, PATCH)
            auth: Optional auth dict with 'type' and 'value' keys.
                  type: 'basic', 'bearer', 'apikey'.
                  For basic: value is 'username:password'.
                  For bearer: value is the token string.
                  For apikey: value is the API key; adds 'Authorization: ApiKey <value>'.
            timeout: Optional timeout in seconds (default: 30)
            follow_redirects: Whether to follow redirects (default: True)

        Returns:
            JSON string with status_code, headers, body, and elapsed_time.
        """
        method = method.upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            return json.dumps({
                "status": "error",
                "error": f"Unsupported HTTP method: {method}. Use GET, POST, PUT, PATCH, DELETE, HEAD, or OPTIONS.",
            }, indent=2)

        timeout = timeout or 30

        # Build headers
        request_headers = dict(headers) if headers else {}

        # Build auth header
        if auth:
            auth_type = auth.get("type", "").lower()
            auth_value = auth.get("value", "")
            if auth_type == "basic":
                import base64
                encoded = base64.b64encode(auth_value.encode()).decode()
                request_headers["Authorization"] = f"Basic {encoded}"
            elif auth_type == "bearer":
                request_headers["Authorization"] = f"Bearer {auth_value}"
            elif auth_type == "apikey":
                request_headers["Authorization"] = f"ApiKey {auth_value}"
            else:
                return json.dumps({
                    "status": "error",
                    "error": f"Unsupported auth type: {auth_type}. Use 'basic', 'bearer', or 'apikey'.",
                }, indent=2)

        # Build request body
        request_body = body if (body and method in _METHODS_WITH_BODY) else None

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=follow_redirects,
                trust_env=False,
            ) as client:
                start = __import__("time").time()
                response = await client.request(
                    method=method,
                    url=url,
                    headers=request_headers,
                    content=request_body,
                )
                elapsed = __import__("time").time() - start

                # Read response body
                response_body = response.text

                # Truncate very large responses
                max_body_len = 50000
                body_truncated = False
                if len(response_body) > max_body_len:
                    response_body = response_body[:max_body_len]
                    body_truncated = True

                result = {
                    "status": "success",
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response_body,
                    "elapsed_seconds": round(elapsed, 3),
                    "url": str(response.url),
                }
                if body_truncated:
                    result["body_truncated"] = True
                    result["body_note"] = f"Response truncated to {max_body_len} chars (original was larger)"

                return json.dumps(result, indent=2, ensure_ascii=False)

        except httpx.TimeoutException:
            return json.dumps({
                "status": "error",
                "error": f"Request timed out after {timeout}s",
            }, indent=2)
        except httpx.RequestError as e:
            return json.dumps({
                "status": "error",
                "error": f"Request failed: {str(e)}",
            }, indent=2)
        except Exception as e:
            logger.error("HTTP request error: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
            }, indent=2)

    logger.info("Registered http_request tool")
