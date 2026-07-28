"""Content extraction from web pages."""

import logging
import re
from typing import Literal

from bs4 import BeautifulSoup, Tag
from html2text import html2text

from src.config import settings

# Truncation mode controls what gets summarized
# - "always": truncate both main text and code blocks (default)
# - "never": no truncation at all
# - "main_only": truncate main text only, preserve code blocks in full
# - "code_only": truncate code blocks only, preserve main text in full
TruncationMode = Literal["always", "never", "main_only", "code_only"]

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
        self._code_blocks: list[dict] = []

    def _remove_boilerplate(self, soup: BeautifulSoup) -> None:
        """Remove boilerplate elements from the BeautifulSoup tree.

        Removes:
        - Elements matching REMOVE_SELECTORS (script, style, noscript, iframe,
          nav, footer, aside, header, form)
        - Elements with common ad/boilerplate class names (ad, advertisement,
          cookie-banner, sidebar, footer-widget)
        - Elements with very low text-to-tag ratio (more than 3 nested tags
          per 10 chars of text — heuristic for ad blocks)
        """
        # Step 1: Remove structural elements
        for selector in REMOVE_SELECTORS:
            for element in soup.find_all(selector):
                element.decompose()

        # Step 2: Remove elements with common ad/boilerplate class names
        boilerplate_classes = ["ad", "advertisement", "cookie-banner",
                               "sidebar", "footer-widget"]
        for el in soup.find_all(class_=boilerplate_classes):
            el.decompose()

        # Step 3: Remove elements with very low text-to-tag ratio
        # More than 3 nested tags per 10 chars of text is likely boilerplate.
        # Skip top-level structural elements to avoid removing the document root.
        structural_tags = {"html", "head", "body", "title"}
        to_remove = []
        for el in soup.find_all(True):
            if el.name in structural_tags:
                continue
            text = el.get_text(strip=True)
            text_len = len(text)
            # Count descendant elements (not including the element itself)
            nested_count = len(el.find_all(True))
            if nested_count <= 0:
                continue  # leaf element, skip
            if text_len == 0:
                # No text but has nested structure — likely boilerplate
                to_remove.append(el)
                continue
            # Check ratio: more than 3 tags per 10 chars means tag_count / text_len > 0.3
            if nested_count / text_len > 3 / 10:
                to_remove.append(el)

        for el in to_remove:
            el.decompose()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate the number of tokens in the given text.

        Uses a heuristic of ~4 characters per token for English text.
        Rounds up conservatively so that the estimate errs on the side
        of truncating more rather than less.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count (conservative upper bound).
        """
        import math
        return math.ceil(len(text) / 4)

    def _find_main_content(self) -> Tag:
        """Find the main content element in the page.

        Strategy:
            1. Look for <article> tag first (most semantic indicator)
            2. Then <main> tag
            3. Then the <body> itself if neither exists
            4. As a fallback heuristic: find the element with the most text content
               (largest get_text() length)

        Returns:
            The BeautifulSoup Tag representing the main content element.
        """
        if self._soup is None:
            raise RuntimeError("No HTML parsed yet — call extract() or set _soup first")

        # 1. Look for <article>
        article = self._soup.find("article")
        if article:
            return article

        # 2. Look for <main>
        main = self._soup.find("main")
        if main:
            return main

        # 3. Fall back to <body>
        body = self._soup.find("body")
        if body:
            return body

        # 4. Fallback heuristic: element with the most text content
        best = self._soup
        best_text_len = len(self._soup.get_text())
        for tag in self._soup.find_all():
            text_len = len(tag.get_text())
            if text_len > best_text_len:
                best = tag
                best_text_len = text_len
        return best

    @staticmethod
    def _extract_code_blocks(text: str, max_chars: int | None) -> list[dict]:
        """Extract fenced code blocks from text.

        Finds all ```...``` blocks, captures the optional language
        identifier and the block content. Each block's content is capped
        at ``max_chars`` with a ``[truncated]`` indicator if exceeded.
        If ``max_chars`` is ``None``, code blocks are preserved in full.

        Args:
            text: The text to extract code blocks from.
            max_chars: Maximum characters per block's content. ``None`` disables truncation.

        Returns:
            List of dicts with ``language`` (str, may be empty) and
            ``content`` (str, possibly truncated) keys.
        """
        pattern = r"```(\w*)\n(.*?)```"
        blocks: list[dict] = []
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            language = match.group(1)
            content = match.group(2).strip()
            if max_chars is not None and len(content) > max_chars:
                content = content[:max_chars] + "[truncated]"
            blocks.append({"language": language, "content": content})
        return blocks

    def extract(
        self,
        html: str,
        max_length: int = 4000,
        truncate: TruncationMode | bool = "always",
        code_block_max_chars: int | None = None,
    ) -> dict:
        """Extract content from HTML.

        Args:
            html: The HTML content to extract from.
            max_length: Maximum total text length before summarization is applied.
            truncate: Truncation mode. One of:
                - "always": truncate both main text and code blocks (default)
                - "never": no truncation at all
                - "main_only": truncate main text only, preserve code blocks in full
                - "code_only": truncate code blocks only, preserve main text in full
                - True/False: backward-compatible aliases for "always"/"never"
            code_block_max_chars: Override max chars per code block. ``None`` uses config default.

        Returns:
            A dictionary containing extracted content with keys:
                - title: Page title
                - content: Main text content (summarized if exceeding max_length)
                - links: List of links
                - images: List of images
                - headings: Structured headings
                - tables: Structured tables
        """
        # Normalize bool to TruncationMode
        if truncate is True:
            mode: TruncationMode = "always"
        elif truncate is False:
            mode = "never"
        else:
            mode = truncate

        # Determine code block truncation settings
        if mode in ("always", "code_only"):
            cb_max = (
                code_block_max_chars
                if code_block_max_chars is not None
                else settings.CODE_BLOCK_MAX_CHARS
            )
        else:
            cb_max = None  # no truncation for code blocks

        # Determine whether to summarize main text
        summarize_main = mode in ("always", "main_only")

        self._soup = BeautifulSoup(html, "html.parser")
        result = {
            "title": self._extract_title(),
            "content": self._extract_text(cb_max),
            "links": self._extract_links(),
            "images": self._extract_images(),
            "headings": self._extract_headings(),
            "tables": self._extract_tables(),
        }

        # Check if summarization is needed
        if summarize_main:
            total_text = len(result["content"])
            if total_text > max_length:
                result["content"] = self._summarize(
                    result["content"],
                    result["headings"],
                    max_length,
                    total_text,
                    token_budget=settings.FETCH_TOKEN_BUDGET,
                )

        return result

    def _summarize(
        self,
        content: str,
        headings: list[dict],
        max_length: int,
        original_length: int,
        token_budget: int | None = None,
    ) -> str:
        """Summarize content when it exceeds max_length.

        When token_budget is provided, uses a three-phase progressive strategy:
            Phase 1 — Budget check: if full content (with code blocks) fits
                within the token budget, return it in full.
            Phase 2 — Section-aware: split content by headings, keep full
                sections that fit their proportional budget share, excerpt
                those that don't, and reinsert code blocks at the end.
            Phase 3 — Hard truncation: if the assembled result still exceeds
                the budget, hard-truncate at the last paragraph boundary.

        When token_budget is None, falls back to char-based summarization
        (original behavior).

        Args:
            content: The full extracted text content.
            headings: List of heading dicts with 'level' and 'text' keys.
            max_length: Target maximum length (char-based fallback).
            original_length: Length of the original content (char-based fallback).
            token_budget: Optional token budget for token-based summarization.

        Returns:
            Summarized content string.
        """
        if token_budget is not None:
            return self._summarize_by_tokens(content, headings, token_budget)

        # --- Char-based fallback (original behavior) ---
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

    def _summarize_by_tokens(self, content: str, headings: list[dict], token_budget: int) -> str:
        """Summarize content using token-based progressive strategy.

        Phase 1: Budget check — return full content if it fits.
        Phase 2: Section-aware — split by headings, keep/excerpt sections.
        Phase 3: Hard truncation — truncate text at paragraph boundary.

        Code blocks are preserved by reserving their token cost upfront
        so that Phases 2 and 3 only operate on the text budget.

        Args:
            content: The full extracted text content (code blocks stripped).
            headings: List of heading dicts with 'level' and 'text' keys.
            token_budget: Maximum token budget for the summarized output.

        Returns:
            Summarized content string with header.
        """
        # Compute original token count (text + code block contents)
        original_tokens = self._estimate_tokens(content) + sum(
            self._estimate_tokens(cb["content"]) for cb in self._code_blocks
        )

        # Reconstruct full content with code blocks appended
        full_content = self._reconstruct_with_code_blocks(content)

        # Phase 1: Budget check — if full content fits, return it
        if self._estimate_tokens(full_content) <= token_budget:
            return full_content

        # Compute header tokens
        header = (
            f"[Summarized — original was {original_tokens} tokens, "
            f"budget {token_budget} tokens]\n"
        )
        header_tokens = self._estimate_tokens(header)

        # Compute code block overhead (separator + formatted blocks)
        code_block_tokens = 0
        if self._code_blocks:
            code_block_tokens += self._estimate_tokens("\n---\nCode blocks:\n")
            for cb in self._code_blocks:
                code_block_tokens += self._estimate_tokens(
                    f"```{cb['language']}\n{cb['content']}\n```"
                )

        # Text budget: total minus header and code blocks
        text_budget = token_budget - header_tokens - code_block_tokens

        # Phase 2: Section-aware summarization (text only)
        sections = self._split_content_by_headings(content, headings)
        total_section_tokens = sum(
            self._estimate_tokens(s["text"]) for s in sections
        )

        text_parts: list[str] = []
        remaining_budget = text_budget

        for section in sections:
            section_tokens = self._estimate_tokens(section["text"])

            # Proportional share of remaining budget
            if total_section_tokens > 0:
                proportional_share = (section_tokens / total_section_tokens) * remaining_budget
            else:
                proportional_share = remaining_budget

            if section_tokens <= proportional_share:
                # Keep full section
                text_parts.append(section["text"])
                remaining_budget -= section_tokens
            else:
                # Excerpt: take first ~50% of section
                excerpt = self._excerpt_section_text(section)
                excerpt_tokens = self._estimate_tokens(excerpt)
                text_parts.append(excerpt)
                remaining_budget -= excerpt_tokens

        text_result = "\n".join(text_parts)

        # Check if text fits within text_budget
        text_tokens = self._estimate_tokens(text_result)
        if text_tokens <= text_budget:
            # Assemble: header + text + code blocks
            if self._code_blocks:
                text_result += "\n---\nCode blocks:\n"
                for cb in self._code_blocks:
                    lang = cb["language"]
                    text_result += f"```{lang}\n{cb['content']}\n```\n"
            return header + text_result

        # Phase 3: Hard truncation of text only (preserves code blocks)
        truncated_text = self._hard_truncate_to_budget(text_result, text_budget)

        # Assemble: header + truncated text + code blocks
        if self._code_blocks:
            truncated_text += "\n---\nCode blocks:\n"
            for cb in self._code_blocks:
                lang = cb["language"]
                truncated_text += f"```{lang}\n{cb['content']}\n```\n"

        return header + truncated_text

    def _reconstruct_with_code_blocks(self, content: str) -> str:
        """Reconstruct content with code blocks appended at the end.

        Args:
            content: Text content with code blocks stripped.

        Returns:
            Content with code blocks reinserted after a separator.
        """
        if not self._code_blocks:
            return content

        parts = [content, "", "---", "Code blocks:"]
        for cb in self._code_blocks:
            lang = cb["language"]
            parts.append(f"```{lang}")
            parts.append(cb["content"])
            parts.append("```")

        return "\n".join(parts)

    def _split_content_by_headings(self, content: str, headings: list[dict]) -> list[dict]:
        """Split content into sections based on heading positions.

        Searches for headings sequentially to ensure correct ordering
        and avoid matching heading text that appears in body content.

        Args:
            content: Text content.
            headings: List of heading dicts with 'level' and 'text' keys.

        Returns:
            List of section dicts with 'heading', 'text', and 'level' keys.
            If no headings are found, returns a single section with the
            full content.
        """
        sections: list[dict] = []
        search_start = 0
        positions: list[tuple[int, dict]] = []

        for heading in headings:
            idx = content.find(heading["text"], search_start)
            if idx != -1:
                positions.append((idx, heading))
                search_start = idx + len(heading["text"])

        # Pre-heading content
        if positions:
            pre_text = content[:positions[0][0]].strip()
            if pre_text:
                sections.append({"heading": None, "text": pre_text, "level": 0})

        # Sections between headings
        for i, (pos, heading) in enumerate(positions):
            next_pos = positions[i + 1][0] if i + 1 < len(positions) else len(content)
            section_text = content[pos:next_pos].strip()
            sections.append({
                "heading": heading["text"],
                "text": section_text,
                "level": heading["level"],
            })

        # If no headings found, treat entire content as one section
        if not sections:
            sections.append({"heading": None, "text": content.strip(), "level": 0})

        return sections

    def _excerpt_section_text(self, section: dict) -> str:
        """Extract an excerpt from a section (up to ~50% of its length).

        Takes the first N lines up to approximately half the section's
        character length. The heading line is naturally included as the
        first line of the section text.

        Args:
            section: Section dict with 'heading', 'text', and 'level' keys.

        Returns:
            Excerpt text with [truncated] indicator if shortened.
        """
        text = section["text"]
        lines = [line for line in text.split("\n") if line.strip()]

        target_len = len(text) // 2
        current_len = 0
        excerpt_lines: list[str] = []

        for line in lines:
            if current_len + len(line) > target_len and current_len > 0:
                break
            excerpt_lines.append(line)
            current_len += len(line)

        if current_len < len(text):
            excerpt_lines.append("[truncated]")

        return "\n".join(excerpt_lines)

    def _extract_sections(
        self,
        content: str,
        headings: list[dict],
        section_names: list[str],
    ) -> str:
        """Extract only the specified sections from content.

        Splits content by headings, keeps only the sections whose heading
        text matches one of the requested section names. Pre-heading content
        (before the first heading) is included if any section is requested.

        Args:
            content: Full extracted text content.
            headings: List of heading dicts with 'level' and 'text' keys.
            section_names: List of heading texts to include.

        Returns:
            Text containing only the requested sections.
        """
        sections = self._split_content_by_headings(content, headings)

        # Build a set of requested heading texts (case-insensitive)
        requested = {name.lower().strip() for name in section_names}

        parts: list[str] = []
        for section in sections:
            heading = section["heading"]
            if heading is None:
                # Pre-heading content: include it if any section is requested
                if requested and section["text"].strip():
                    parts.append(section["text"].strip())
            elif heading.lower().strip() in requested:
                # Reconstruct heading with markdown prefix
                level = section["level"]
                prefix = "#" * level
                parts.append(f"{prefix} {heading}")
                parts.append(section["text"])  # includes heading line already

        if not parts:
            # Fallback: if no sections matched, return original content
            return content

        return "\n\n".join(parts)

    def _hard_truncate_to_budget(self, text: str, token_budget: int) -> str:
        """Hard-truncate text at the last paragraph boundary to fit within budget.

        Uses a conservative estimate of 3.5 chars per token to ensure
        the result stays within the token budget even after adding the
        [truncated] indicator.

        Args:
            text: Text to truncate.
            token_budget: Maximum token budget.

        Returns:
            Truncated text with [truncated] indicator.
        """
        if self._estimate_tokens(text) <= token_budget:
            return text

        # Conservative target: 3.5 chars per token (leaves room for [truncated])
        target_chars = int(token_budget * 3.5) - 20

        truncated = text[:target_chars]
        # Find last paragraph boundary (newline)
        last_newline = truncated.rfind("\n")
        if last_newline > target_chars // 2:
            truncated = truncated[:last_newline]

        return truncated.rstrip() + "\n\n[truncated]"

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

    def _extract_text(self, code_block_max_chars: int | None = None) -> str:
        """Extract text content from the page.

        Pipeline:
            1. Remove boilerplate from soup
            2. Find main content area
            3. Extract text from main area
            4. Extract code blocks and store in self._code_blocks
            5. Strip code blocks from text body
            6. Clean up whitespace

        Returns:
            Cleaned text content with code blocks removed from body.
        """
        if self._soup is None:
            return ""

        self._code_blocks = []

        # Step 1: Remove boilerplate
        self._remove_boilerplate(self._soup)

        # Step 2: Find main content area
        main = self._find_main_content()

        # Step 3: Extract text from main area
        text = main.get_text(separator="\n", strip=True)

        # Step 4: Extract code blocks and store
        self._code_blocks = self._extract_code_blocks(text, max_chars=4000)

        # Step 5: Strip code blocks from text body
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

        # Step 6: Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        result = "\n".join(lines)

        # Step 7: Re-insert code blocks at the end
        if self._code_blocks:
            code_text = "\n\n".join(
                f"```{b['language']}\n{b['content']}\n```"
                for b in self._code_blocks
            )
            result = result + "\n\n" + code_text if result else code_text

        return result

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
