"""File operations tools for MCP server."""

import base64
import json
import logging
from pathlib import Path

from mcp.server import FastMCP

from src.config import settings

logger = logging.getLogger(__name__)


def file_operations_handler(server: FastMCP) -> None:
    """Register file read, list, and delete tools."""

    @server.tool()
    async def file_read(filename: str) -> str:
        """Read a file from the output directory. Binary files returned as base64."""
        # Validate filename
        if "/" in filename or ".." in filename:
            return json.dumps({
                "status": "error",
                "error": "Filename must not contain path separators or '..'",
            }, indent=2)

        file_path = Path(settings.FILE_OUTPUT_DIR) / filename
        if not file_path.exists():
            return json.dumps({
                "status": "error",
                "error": f"File not found: {filename}",
            }, indent=2)
        if not file_path.is_file():
            return json.dumps({
                "status": "error",
                "error": f"Not a file: {filename}",
            }, indent=2)

        try:
            # Try text first
            content = file_path.read_text(encoding="utf-8")
            file_size = file_path.stat().st_size
            return json.dumps({
                "status": "success",
                "filename": filename,
                "content": content,
                "size": file_size,
                "encoding": "utf-8",
            }, indent=2, ensure_ascii=False)
        except UnicodeDecodeError:
            # Binary file — return as base64
            import base64
            raw = file_path.read_bytes()
            file_size = file_path.stat().st_size
            return json.dumps({
                "status": "success",
                "filename": filename,
                "content": base64.b64encode(raw).decode("utf-8"),
                "size": file_size,
                "encoding": "base64",
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to read file: {str(e)}",
            }, indent=2)

    @server.tool()
    async def file_list(directory: str = "") -> str:
        """List files in the output directory or a subdirectory."""
        # Validate directory
        if "/" in directory or ".." in directory:
            return json.dumps({
                "status": "error",
                "error": "Directory must not contain path separators or '..'",
            }, indent=2)

        base = Path(settings.FILE_OUTPUT_DIR)
        if directory:
            target = base / directory
            if not target.exists():
                return json.dumps({
                    "status": "error",
                    "error": f"Directory not found: {directory}",
                }, indent=2)
            if not target.is_dir():
                return json.dumps({
                    "status": "error",
                    "error": f"Not a directory: {directory}",
                }, indent=2)
        else:
            target = base

        try:
            entries = []
            for entry in sorted(target.iterdir()):
                is_dir = entry.is_dir()
                size = entry.stat().st_size if not is_dir else 0
                entries.append({
                    "name": entry.name,
                    "type": "directory" if is_dir else "file",
                    "size": size,
                    "modified": str(entry.stat().st_mtime),
                })
            return json.dumps({
                "status": "success",
                "directory": directory or "/",
                "entries": entries,
                "total": len(entries),
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to list directory: {str(e)}",
            }, indent=2)

    @server.tool()
    async def file_delete(filename: str) -> str:
        """Delete a file from the output directory."""
        # Validate filename
        if "/" in filename or ".." in filename:
            return json.dumps({
                "status": "error",
                "error": "Filename must not contain path separators or '..'",
            }, indent=2)

        file_path = Path(settings.FILE_OUTPUT_DIR) / filename

        if not file_path.exists():
            return json.dumps({
                "status": "error",
                "error": f"File not found: {filename}",
            }, indent=2)
        if not file_path.is_file():
            return json.dumps({
                "status": "error",
                "error": f"Not a file (cannot delete directories): {filename}",
            }, indent=2)

        try:
            file_path.unlink()
            return json.dumps({
                "status": "success",
                "message": f"Deleted file: {filename}",
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to delete file: {str(e)}",
            }, indent=2)

    @server.tool()
    async def file_upload(filename: str, content: str, encoding: str = "utf-8") -> str:
        """Upload a file to the output directory. encoding: 'utf-8' (default) or 'base64' for binary."""
        # Validate filename
        if "/" in filename or ".." in filename:
            return json.dumps({
                "status": "error",
                "error": "Filename must not contain path separators or '..'",
            }, indent=2)

        file_path = Path(settings.FILE_OUTPUT_DIR) / filename

        try:
            if encoding == "base64":
                raw = base64.b64decode(content)
                file_path.write_bytes(raw)
            else:
                file_path.write_text(content, encoding="utf-8")

            file_size = file_path.stat().st_size
            download_url = f"{settings.FILE_BASE_URL}/files/{filename}"
            return json.dumps({
                "status": "success",
                "filename": filename,
                "size": file_size,
                "download_url": download_url,
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": f"Failed to write file: {str(e)}",
            }, indent=2)

    logger.info("Registered file operations tools")
