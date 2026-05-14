"""Content extraction from web pages."""

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup
from html2text import html2text

logger = logging.getLogger(__name__)

# Elements to remove during cleaning
REMOVE_SELECTORS = ["script", "style", "noscript", "iframe", "nav", "footer", "aside", "header", "form"]

# Elements considered as content
CONTENT_SELECTORS = ["article", "main", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
                     "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td", "blockquote",
                     "figure", "figcaption", "pre", "code", "a", "img", "span"]


class ContentExtractor:
    """Extracts content from HTML pages."""

    def __init__(self):
        """Initialize the content extractor."""
        self._soup = None

    def extract(self, html: str, max_length: int = 4000, truncate: bool = True) -> dict:
        """Extract content from HTML.

        Args:
            html: The HTML content to extract from.
            max_length: Maximum total text length before summarization is applied.
            truncate: When True, summarize content that exceeds max_length. When False, return full content.

        Returns:
            A dictionary containing extracted content with keys:
                - title: Page title
                - content: Main text content (summarized if it exceeds max_length and truncate is True)
                - links: List of links
                - images: List of images
                - headings: Structured headings
                - tables: Structured tables
        """
        self._soup = BeautifulSoup(html, "html.parser")
        result = {
            "title": self._extract_title(),
            "content": self._extract_text(),
            "links": self._extract_links(),
            "images": self._extract_images(),
            "headings": self._extract_headings(),
            "tables": self._extract_tables(),
        }

        # Check if summarization is needed
        if truncate:
            total_text = len(result["content"])
            if total_text > max_length:
                result["content"] = self._summarize(
                    result["content"],
                    result["headings"],
                    max_length,
                    total_text,
                )

        return result

    def _summarize(self, content: str, headings: list[dict], max_length: int, original_length: int) -> str:
        """Summarize content when it exceeds max_length.

        Strategy:
            1. Always keep the first ~200 chars (first meaningful paragraph)
            2. Use headings to identify sections and keep ~100-150 chars per section
            3. Remove code blocks and table content first
            4. Add a summary indicator

        Args:
            content: The full extracted text content.
            headings: List of heading dicts with 'level' and 'text' keys.
            max_length: Target maximum length.
            original_length: Length of the original content.

        Returns:
            Summarized content string.
        """
        # Strip code blocks (between ``` markers)
        cleaned = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

        # Split into lines for processing
        lines = cleaned.split("\n")

        # Build summary: always keep first ~200 chars
        summary_parts: list[str] = []
        summary_parts.append(f"[Summarized — original was {original_length} chars, showing key sections]\n")

        # First paragraph: first ~200 chars of meaningful text
        first_para = ""
        for line in lines:
            stripped = line.strip()
            if stripped:
                first_para += stripped + "\n"
        first_para = first_para.strip()
        if len(first_para) > 200:
            first_para = first_para[:200].rsplit(" ", 1)[0] + "..."
        if first_para:
            summary_parts.append(first_para)

        # If we have headings, use them to extract section excerpts
        if headings:
            summary_parts.append("")  # blank line separator
            for heading in headings:
                section_text = self._extract_section_excerpt(content, heading["text"])
                if section_text:
                    level = heading["level"]
                    prefix = "#" * level
                    summary_parts.append(f"{prefix} {heading['text']}")
                    summary_parts.append(section_text)

        # Assemble and check length
        result = "\n".join(summary_parts)

        # If still too long, hard-truncate with ellipsis
        if len(result) > max_length:
            result = result[:max_length - 3].rsplit(" ", 1)[0] + "..."

        return result

    def _extract_section_excerpt(self, content: str, heading_text: str, excerpt_len: int = 120) -> str:
        """Extract a brief excerpt from the section following a heading.

        Args:
            content: Full content text.
            heading_text: The heading text to search for.
            excerpt_len: Target excerpt length in characters.

        Returns:
            Brief excerpt text, or empty string if not found.
        """
        # Find the heading in the content (it may appear as-is or with markdown prefix)
        idx = content.find(heading_text)
        if idx == -1:
            return ""

        # Start after the heading
        start = idx + len(heading_text)
        # Take the next ~excerpt_len chars, stripping code blocks
        excerpt = content[start:start + excerpt_len + 200]
        # Remove code blocks from excerpt
        excerpt = re.sub(r"```.*?```", "", excerpt, flags=re.DOTALL)
        excerpt = excerpt.strip()

        if len(excerpt) > excerpt_len:
            excerpt = excerpt[:excerpt_len].rsplit(" ", 1)[0] + "..."

        return excerpt

    def _extract_title(self) -> str:
        """Extract the page title.

        Returns:
            The page title as a string.
        """
        if self._soup is None:
            return ""

        title = self._soup.find("title")
        if title:
            return title.get_text(strip=True)

        h1 = self._soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        return ""

    def _extract_text(self) -> str:
        """Extract text content from the page.

        Returns:
            Cleaned text content.
        """
        if self._soup is None:
            return ""

        # Remove script and style elements
        for element in self._soup.find_all(["script", "style", "noscript"]):
            element.decompose()

        # Get text content
        text = self._soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    def _extract_links(self) -> list[dict]:
        """Extract links from the page.

        Returns:
            List of dictionaries with 'text' and 'url' keys.
        """
        if self._soup is None:
            return []

        links = []
        for a in self._soup.find_all("a", href=True):
            links.append({
                "text": a.get_text(strip=True),
                "url": a["href"],
            })

        logger.info("Extracted %d links", len(links))
        return links

    def _extract_images(self) -> list[dict]:
        """Extract images from the page.

        Returns:
            List of dictionaries with 'alt' and 'src' keys.
        """
        if self._soup is None:
            return []

        images = []
        for img in self._soup.find_all("img", src=True):
            images.append({
                "alt": img.get("alt", ""),
                "src": img["src"],
            })

        logger.info("Extracted %d images", len(images))
        return images

    def _extract_headings(self) -> list[dict]:
        """Extract headings from the page in a structured format.

        Returns:
            List of dictionaries with 'level', 'text', and 'id' keys.
        """
        if self._soup is None:
            return []

        headings = []
        for heading in self._soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(heading.name[1])  # h1 -> 1, h2 -> 2, etc.
            text = heading.get_text(strip=True)
            heading_id = heading.get("id", "")
            headings.append({
                "level": level,
                "text": text,
                "id": heading_id,
            })

        logger.info("Extracted %d headings", len(headings))
        return headings

    def _extract_tables(self) -> list[list[list[str]]]:
        """Extract tables from the page.

        Returns:
            List of tables, each table is a list of rows, each row is a list of cell values.
        """
        if self._soup is None:
            return []

        tables = []
        for table in self._soup.find_all("table"):
            rows = []
            for tr in table.find_all(["tr", "th", "td"]):
                row = []
                for cell in tr.find_all(["td", "th"]):
                    row.append(cell.get_text(strip=True))
                if row:
                    rows.append(row)
            if rows:
                tables.append(rows)

        logger.info("Extracted %d tables", len(tables))
        return tables

    def extract_to_markdown(self, html: str) -> str:
        """Extract content and convert to markdown.

        Args:
            html: The HTML content to extract from.

        Returns:
            Markdown formatted content.
        """
        content = self.extract(html)
        return html2text(content["content"])

    def get_clean_html(self, html: str) -> str:
        """Get clean HTML with only the main content.

        Args:
            html: The HTML content to clean.

        Returns:
            Cleaned HTML content.
        """
        if self._soup is None:
            self._soup = BeautifulSoup(html, "html.parser")

        # Remove common non-content elements
        for selector in REMOVE_SELECTORS:
            for element in self._soup.find_all(selector):
                element.decompose()

        return str(self._soup)

    def to_llm_format(self, html: str) -> str:
        """Extract content and format it for LLM consumption.

        Creates a structured, LLM-friendly format with headings, content,
        links, and tables clearly separated.

        Args:
            html: The HTML content to extract from.

        Returns:
            LLM-friendly formatted string.
        """
        content = self.extract(html)

        parts = []

        # Title
        if content.get("title"):
            parts.append(f"# {content['title']}\n")

        # Headings with content
        headings = content.get("headings", [])
        if headings:
            parts.append("## Headings\n")
            for heading in headings:
                indent = "  " * (heading["level"] - 1)
                parts.append(f"{indent}- {heading['text']}\n")

        # Main content
        text_content = content.get("content", "")
        if text_content:
            parts.append("\n## Content\n")
            parts.append(text_content)

        # Tables
        tables = content.get("tables", [])
        if tables:
            parts.append("\n## Tables\n")
            for i, table in enumerate(tables):
                parts.append(f"\n### Table {i + 1}\n")
                for row in table:
                    parts.append("| " + " | ".join(row) + " |\n")

        # Links
        links = content.get("links", [])
        if links:
            parts.append("\n## Links\n")
            for link in links:
                parts.append(f"- [{link['text']}]({link['url']})\n")

        # Images
        images = content.get("images", [])
        if images:
            parts.append("\n## Images\n")
            for image in images:
                alt = image.get("alt", "image")
                parts.append(f"- ![{alt}]({image['src']})\n")

        return "\n".join(parts)
