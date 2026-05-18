"""Time and date tool for MCP server."""

import json
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any

from mcp.server import FastMCP

logger = logging.getLogger(__name__)

# Common timezone aliases
_TIMEZONE_ALIASES = {
    "utc": "UTC",
    "gmt": "UTC",
    "est": "America/New_York",
    "edt": "America/New_York",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "gmt": "UTC",
    "cet": "Europe/Berlin",
    "cest": "Europe/Berlin",
    "ist": "Asia/Kolkata",
    "cst_china": "Asia/Shanghai",
    "jst": "Asia/Tokyo",
    "kst": "Asia/Seoul",
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
    "npt": "Asia/Kathmandu",
    "idt": "Asia/Jerusalem",
}


def _resolve_timezone(tz_name: str) -> ZoneInfo:
    """Resolve a timezone name to a ZoneInfo object."""
    # Try direct lookup
    try:
        return ZoneInfo(tz_name)
    except Exception:
        pass

    # Try alias lookup
    lower = tz_name.lower()
    if lower in _TIMEZONE_ALIASES:
        try:
            return ZoneInfo(_TIMEZONE_ALIASES[lower])
        except Exception:
            pass

    raise ValueError(f"Unknown timezone: {tz_name}")


def time_now_handler(server: FastMCP) -> None:
    """Register the time_now tool."""

    @server.tool()
    async def time_now(
        timezone_name: str = "UTC",
        format: str | None = None,
        convert_from_timezone: str | None = None,
        convert_from_time: str | None = None,
    ) -> str:
        """Get current time/date or convert between timezones.

        Two modes:
        1. Default: returns current time in the specified timezone
        2. Conversion: convert a time from one timezone to another

        Args:
            timezone_name: Target timezone (default: UTC). Supports IANA names
                           like 'America/New_York', 'Europe/London', 'Asia/Tokyo',
                           or aliases like 'EST', 'PST', 'CET', 'IST', 'JST'.
            format: Optional strftime format string (e.g., '%Y-%m-%d %H:%M:%S').
                    If not specified, returns ISO format.
            convert_from_timezone: Source timezone for conversion.
            convert_from_time: Time string to convert (e.g., '2026-01-15 14:30:00').
                               Must be provided with convert_from_timezone.

        Returns:
            JSON string with time information.
        """
        try:
            target_tz = _resolve_timezone(timezone_name)

            # Conversion mode
            if convert_from_timezone and convert_from_time:
                source_tz = _resolve_timezone(convert_from_timezone)
                # Parse the source time
                if format:
                    dt = datetime.strptime(convert_from_time, format)
                else:
                    # Try ISO format first, then common formats
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]:
                        try:
                            dt = datetime.strptime(convert_from_time, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        return json.dumps({
                            "status": "error",
                            "error": f"Could not parse time: {convert_from_time}. Use ISO format or specify format parameter.",
                        }, indent=2)

                # Localize to source timezone
                dt = dt.replace(tzinfo=source_tz)
                # Convert to target timezone
                dt_converted = dt.astimezone(target_tz)

                result = {
                    "status": "success",
                    "operation": "conversion",
                    "source_time": convert_from_time,
                    "source_timezone": convert_from_timezone,
                    "converted_time": dt_converted.isoformat(),
                    "target_timezone": timezone_name,
                }
                if format:
                    result["formatted"] = dt_converted.strftime(format)

                return json.dumps(result, indent=2)

            # Default mode: current time
            now = datetime.now(timezone.utc).astimezone(target_tz)

            if format:
                formatted = now.strftime(format)
            else:
                formatted = now.isoformat()

            result = {
                "status": "success",
                "operation": "current_time",
                "iso_format": now.isoformat(),
                "formatted": formatted,
                "timezone": timezone_name,
                "utc_offset": str(now.utcoffset()),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "unix_timestamp": now.timestamp(),
            }

            return json.dumps(result, indent=2)

        except ValueError as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
            }, indent=2)
        except Exception as e:
            logger.error("Time now error: %s", str(e))
            return json.dumps({
                "status": "error",
                "error": str(e),
            }, indent=2)

    logger.info("Registered time_now tool")
