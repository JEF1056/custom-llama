"""Current time/date tool for MCP server."""

import json
import logging
from datetime import datetime, timezone
from typing import Annotated
from zoneinfo import ZoneInfo

from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import Field

logger = logging.getLogger(__name__)

# Default timezone when the caller doesn't specify one.
_DEFAULT_TZ = "America/Los_Angeles"  # Pacific (PST/PDT)

# Common timezone aliases → IANA names. DST is handled automatically.
_TIMEZONE_ALIASES = {
    "utc": "UTC",
    "gmt": "UTC",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "pt": "America/Los_Angeles",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "mt": "America/Denver",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "ct": "America/Chicago",
    "est": "America/New_York",
    "edt": "America/New_York",
    "et": "America/New_York",
    "cet": "Europe/Berlin",
    "cest": "Europe/Berlin",
    "bst": "Europe/London",
    "ist": "Asia/Kolkata",
    "jst": "Asia/Tokyo",
    "kst": "Asia/Seoul",
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
}


def _resolve_timezone(tz_name: str) -> ZoneInfo:
    """Resolve an IANA name or common alias to a ZoneInfo object.

    Aliases are checked first: civil abbreviations like EST/PST should map to the
    DST-aware regional zone (e.g. America/New_York), NOT the fixed-offset IANA
    zones of the same name (ZoneInfo("EST") is a permanent -0500 with no DST).
    """
    alias = _TIMEZONE_ALIASES.get(tz_name.strip().lower())
    if alias:
        return ZoneInfo(alias)

    try:
        return ZoneInfo(tz_name)
    except Exception:
        raise ValueError(
            f"Unknown timezone: {tz_name!r}. Use an IANA name (e.g. America/New_York) "
            "or an alias (PST, EST, UTC, JST)."
        )


def time_now_handler(server: FastMCP) -> None:
    """Register the time_now tool."""

    @server.tool()
    async def time_now(
        timezone_name: Annotated[
            str,
            Field(description="IANA name (e.g. America/New_York) or alias (PST, EST, UTC, JST). Defaults to Pacific."),
        ] = _DEFAULT_TZ,
        ctx: Context | None = None,
    ) -> str:
        """Get the current date and time in a timezone (default: PST/Pacific).

        timezone_name: IANA name (America/New_York) or alias (PST, EST, UTC, JST).
        Returns a detailed breakdown: human-readable string, date, 24h/12h time,
        day of week, day of year, week number, ISO 8601, timezone abbreviation,
        UTC offset, and unix timestamp.
        """
        try:
            if ctx:
                await ctx.report_progress(0, 1, "Resolving timezone\u2026")
            tz = _resolve_timezone(timezone_name)
            now = datetime.now(timezone.utc).astimezone(tz)
            if ctx:
                await ctx.report_progress(1, 1, "Done")
            return json.dumps({
                "status": "success",
                "timezone": timezone_name,
                "datetime": now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z"),
                "date": now.strftime("%Y-%m-%d"),
                "time_24h": now.strftime("%H:%M:%S"),
                "time_12h": now.strftime("%I:%M:%S %p").lstrip("0"),
                "day_of_week": now.strftime("%A"),
                "month": now.strftime("%B"),
                "day": now.day,
                "year": now.year,
                "day_of_year": int(now.strftime("%j")),
                "week_of_year": int(now.strftime("%V")),
                "iso": now.isoformat(),
                "tz_abbreviation": now.strftime("%Z"),
                "utc_offset": now.strftime("%z"),
                "unix_timestamp": int(now.timestamp()),
            }, indent=2)
        except ValueError as e:
            return json.dumps({"status": "error", "error": str(e)}, indent=2)
        except Exception as e:
            logger.error("Time now error: %s", str(e))
            return json.dumps({"status": "error", "error": str(e)}, indent=2)

    logger.info("Registered time_now tool")
