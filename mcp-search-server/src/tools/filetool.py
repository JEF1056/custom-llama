"""File creation tool for MCP server."""

import base64
import logging
from pathlib import Path

from mcp.server import FastMCP
from mcp.types import EmbeddedResource, TextContent

from src.config import settings

logger = logging.getLogger(__name__)

# Mapping of format names to (extension, content_type)
FORMAT_MAP = {
    "pdf":  (".pdf",  "application/pdf"),
    "svg":  (".svg",  "image/svg+xml"),
    "html": (".html", "text/html"),
    "json": (".json", "application/json"),
    "csv":  (".csv",  "text/csv"),
    "xml":  (".xml",  "application/xml"),
}


def _wrap_pdf(content: str) -> str:
    """Wrap LLM-written PDF content in a minimal valid PDF structure.

    If the content already starts with %PDF-, it is returned as-is.
    Otherwise a one-page PDF shell is built around the content,
    treating it as the stream data for the page's content object.
    """
    if content.startswith("%PDF-"):
        return content

    # Escape special PDF operators in the content
    escaped = content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    body = f"BT /F1 12 Tf 50 750 Td ({escaped}) Tj ET"
    body_len = len(body)

    # Build objects and compute offsets
    obj1 = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    obj4 = f"4 0 obj\n<< /Length {body_len} >>\nstream\n{body}\nendstream\nendobj\n"
    obj5 = "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    header = "%PDF-1.4\n"
    body_text = obj1 + obj2 + obj3 + obj4 + obj5
    xref_offset = len(header) + len(body_text)

    xref = (
        "xref\n0 6\n"
        "0000000000 65535 f \n"
        f"{0:010d} 00000 n \n"
        f"{len(header):010d} 00000 n \n"
        f"{len(header) + len(obj1):010d} 00000 n \n"
        f"{len(header) + len(obj1) + len(obj2):010d} 00000 n \n"
        f"{len(header) + len(obj1) + len(obj2) + len(obj3):010d} 00000 n \n"
        f"{len(header) + len(obj1) + len(obj2) + len(obj3) + len(obj4):010d} 00000 n \n"
    )

    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"

    return header + body_text + xref + trailer


def _resolve_filename(filename: str, fmt: str | None) -> str:
    """Append the correct extension if the filename lacks one and a format is given."""
    if fmt and fmt in FORMAT_MAP:
        expected_ext = FORMAT_MAP[fmt][0]
        if not filename.lower().endswith(expected_ext):
            return filename + expected_ext
    return filename


def create_file_handler(server: FastMCP) -> None:
    """Register the create_file tool."""

    @server.tool()
    async def create_file(
        filename: str,
        content: str,
        content_type: str = "text/plain",
        encoding: str = "utf-8",
        format: str | None = None,
    ):
        """Create a file. Returns MCP EmbeddedResource + resource URI.

        format: pdf | svg | html | json | csv | xml — auto-sets extension and content_type.
        PDF: write BT/ET commands; tool wraps structure automatically.
        encoding: 'utf-8' (default) or 'base64' for binary content.
        """
        try:
            # --- Resolve format ------------------------------------------------
            if format and format not in FORMAT_MAP:
                return [TextContent(type="text", text=f"Error: unsupported format '{format}'. Supported: {', '.join(FORMAT_MAP.keys())}")]

            if format:
                expected_ext, auto_content_type = FORMAT_MAP[format]
                filename = _resolve_filename(filename, format)
                # Auto-set content_type only if the caller used the default
                if content_type == "text/plain":
                    content_type = auto_content_type

            # --- PDF auto-wrap --------------------------------------------------
            if format == "pdf" and encoding == "utf-8":
                content = _wrap_pdf(content)

            # --- Validate encoding parameter ------------------------------------
            if encoding not in ("utf-8", "base64"):
                return [TextContent(type="text", text=f"Error: encoding must be 'utf-8' or 'base64', got '{encoding}'")]

            # --- Validate filename - prevent path traversal ---------------------
            if "/" in filename or ".." in filename:
                return [TextContent(type="text", text="Error: filename must not contain path separators or '..'")]

            # --- Ensure output directory exists ---------------------------------
            output_dir = Path(settings.FILE_OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)

            # --- Resolve the full file path ------------------------------------
            file_path = output_dir / filename

            # --- Write the file content ----------------------------------------
            if encoding == "base64":
                try:
                    raw_bytes = base64.b64decode(content)
                except Exception as e:
                    return [TextContent(type="text", text=f"Error: invalid base64 content - {str(e)}")]

                file_path.write_bytes(raw_bytes)
                # For binary content, embed as base64 blob
                resource_uri = f"file://{file_path}"
                embedded = EmbeddedResource(
                    type="resource",
                    resource={
                        "uri": resource_uri,
                        "mimeType": content_type,
                        "blob": content,  # already base64
                    },
                    annotations={"priority": 1},
                )
            else:
                file_path.write_text(content, encoding="utf-8")
                # For text content, embed as text
                resource_uri = f"file://{file_path}"
                embedded = EmbeddedResource(
                    type="resource",
                    resource={
                        "uri": resource_uri,
                        "mimeType": content_type,
                        "text": content,
                    },
                    annotations={"priority": 1},
                )

            file_size = file_path.stat().st_size

            info_text = (
                f"File created successfully.\n"
                f"  Resource URI: {resource_uri}\n"
                f"  Download URL: {settings.FILE_BASE_URL}/files/{filename}\n"
                f"  Size: {file_size} bytes\n"
                f"  Type: {content_type}\n"
                f"  Encoding: {encoding}\n"
                f"  Format: {format if format else '(none)'}\n"
                f"\n"
                f"The file content is embedded above as an MCP resource. "
                f"You can also access it later via the resource URI or download URL."
            )

            return [embedded, TextContent(type="text", text=info_text)]

        except PermissionError as e:
            logger.error("Permission error creating file %s: %s", filename, str(e))
            return [TextContent(type="text", text=f"Error: permission denied writing to {settings.FILE_OUTPUT_DIR} - {str(e)}")]
        except OSError as e:
            logger.error("OS error creating file %s: %s", filename, str(e))
            return [TextContent(type="text", text=f"Error: failed to create file - {str(e)}")]
        except Exception as e:
            logger.error("Unexpected error creating file %s: %s", filename, str(e))
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    logger.info("Registered create_file tool")
