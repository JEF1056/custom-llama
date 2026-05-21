"""PowerPoint (.pptx) read tool for MCP server."""

import json
import logging
from pathlib import Path

from mcp.server import FastMCP
from mcp.types import TextContent
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from src.config import settings

logger = logging.getLogger(__name__)

# Layout index to name mapping (matches pptx_create.py)
_LAYOUT_NAMES = {
    0: "title",
    1: "title_content",
    2: "section_header",
    3: "two_column",
    6: "blank",
}


def _validate_filename(filename: str) -> str | None:
    """Validate filename for path traversal. Returns error message or None."""
    if not filename:
        return "Error: filename is required"
    if "/" in filename or "\\" in filename or ".." in filename:
        return "Error: filename must not contain path separators or '..'"
    if not filename.lower().endswith(".pptx"):
        return "Error: filename must end with .pptx"
    return None


def _extract_text_from_shape(shape) -> str:
    """Extract all text from a shape that has a text frame."""
    if not shape.has_text_frame:
        return ""
    parts = []
    for paragraph in shape.text_frame.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_table_data(shape) -> dict:
    """Extract table data from a table shape."""
    if not shape.has_table:
        return {}
    table = shape.table
    headers = []
    rows = []
    for row_idx, row in enumerate(table.rows):
        row_data = []
        for cell in row.cells:
            row_data.append(cell.text.strip())
        if row_idx == 0:
            headers = row_data
        else:
            rows.append(row_data)
    return {"headers": headers, "rows": rows}


def _describe_shape(shape) -> dict:
    """Describe a shape's type and content for the read output."""
    info = {"shape_id": shape.shape_id}

    if shape.has_text_frame:
        text = _extract_text_from_shape(shape)
        if text:
            info["text"] = text

    if shape.has_table:
        info["type"] = "table"
        info["table"] = _extract_table_data(shape)
    elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        info["type"] = "image"
        info["image"] = {
            "width": shape.width,
            "height": shape.height,
            "left": shape.left,
            "top": shape.top,
        }
    elif shape.has_text_frame:
        info["type"] = "text"
    else:
        info["type"] = "other"

    return info


def pptx_read_handler(server: FastMCP) -> None:
    """Register the pptx_read tool."""

    @server.tool()
    async def pptx_read(
        filename: str,
        slide_index: int | None = None,
    ):
        """Read an existing PowerPoint (.pptx) presentation and return structured data.

        Loads the presentation from the server's file output directory and returns
        slide structure, content, tables, and image references as JSON.

        **Preferred workflow: create small, then edit**

        Rather than building a complete presentation in one ``pptx_create`` call,
        create a minimal version first (e.g. title slide + one content slide),
        then use ``pptx_edit`` to add slides, fix text, or insert tables.
        This iterative approach is more reliable and easier to debug.

        **Parameters**

        - ``filename`` (str): Name of the .pptx file to read (e.g., "presentation.pptx").
          Must not contain path separators or ``..``.
        - ``slide_index`` (int | None): Specific slide index to read (0-based).
          If ``None``, all slides are returned.

        **Return format**

        Returns a JSON string with:

        - ``status``: ``"success"`` or ``"error"``
        - ``filename``: the requested filename
        - ``slide_count``: total number of slides
        - ``slides``: list of slide objects, each with:
          - ``index``: 0-based slide index
          - ``layout``: layout name (e.g. "title_content", "blank")
          - ``title``: slide title text (if present)
          - ``shapes``: list of shape descriptions with type and content
          - ``notes``: speaker notes text (if present)

        Args:
            filename: Name of the .pptx file to read
            slide_index: Optional specific slide index (0-based) to read

        Returns:
            JSON string with structured presentation data.
        """
        try:
            # --- Validate filename ---------------------------------------------
            err = _validate_filename(filename)
            if err:
                return [TextContent(type="text", text=err)]

            # --- Resolve file path ---------------------------------------------
            output_dir = Path(settings.FILE_OUTPUT_DIR)
            file_path = output_dir / filename

            if not file_path.exists():
                return [TextContent(
                    type="text",
                    text=f"Error: file '{filename}' not found in {settings.FILE_OUTPUT_DIR}",
                )]

            # --- Load presentation ---------------------------------------------
            prs = Presentation(str(file_path))

            # --- Determine which slides to read --------------------------------
            slide_indices = range(len(prs.slides))
            if slide_index is not None:
                if slide_index < 0 or slide_index >= len(prs.slides):
                    return [TextContent(
                        type="text",
                        text=f"Error: slide index {slide_index} out of range (0-{len(prs.slides) - 1})",
                    )]
                slide_indices = [slide_index]

            # --- Read each slide -----------------------------------------------
            slides_data = []
            for idx in slide_indices:
                slide = prs.slides[idx]

                # Determine layout name
                layout_name = "unknown"
                slide_layout = slide.slide_layout
                for layout_idx, name in _LAYOUT_NAMES.items():
                    if layout_idx < len(prs.slide_layouts):
                        if prs.slide_layouts[layout_idx] == slide_layout:
                            layout_name = name
                            break

                # Extract title from title placeholder
                title_text = ""
                if slide.shapes.title and slide.shapes.title.has_text_frame:
                    title_text = slide.shapes.title.text_frame.text.strip()

                # Describe all shapes
                shapes = []
                for shape in slide.shapes:
                    shape_info = _describe_shape(shape)
                    shapes.append(shape_info)

                # Extract speaker notes
                notes_text = ""
                if slide.has_notes_slide and slide.notes_slide:
                    notes_frame = slide.notes_slide.notes_text_frame
                    if notes_frame:
                        notes_text = notes_frame.text.strip()

                slides_data.append({
                    "index": idx,
                    "layout": layout_name,
                    "title": title_text,
                    "shapes": shapes,
                    "notes": notes_text,
                })

            # --- Build response ------------------------------------------------
            result = {
                "status": "success",
                "filename": filename,
                "slide_count": len(prs.slides),
                "slides": slides_data,
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except PermissionError as e:
            logger.error("Permission error reading file %s: %s", filename, str(e))
            msg = f"Error: permission denied reading {filename}"
            return [TextContent(type="text", text=f"{msg} - {str(e)}")]
        except Exception as e:
            logger.error("Unexpected error reading file %s: %s", filename, str(e))
            return [TextContent(type="text", text=f"Error: failed to read '{filename}' - {str(e)}")]

    logger.info("Registered pptx_read tool")
