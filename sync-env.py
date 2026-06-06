#!/usr/bin/env python3
"""Sync .env.default → root .env.

Root .env.default is the single source of truth.  Running this script:
  - Writes / updates root .env (all variables; consumed by docker-compose)

Root .env behaviour:
  - New variables are ADDED (with their default value).
  - Non-secret variables are UPDATED when .env.default changes.
  - Secret variables are PRESERVED (never overwritten after initial write).
  - Stale variables (removed from .env.default) are DELETED.
  - Auto-generated tokens (LLAMA_API_KEY, MCP_API_KEY) are created on first run.

opencode.json is generated from opencode-default.json by substituting {env:VAR}
placeholders from root .env and "{ini:SECTION.KEY}" placeholders from config/models.ini
(e.g. "{ini:qwen3.6-27b.ctx-size}" → the ctx-size value for that model).

Usage:
    python sync-env.py              # sync root .env
    python sync-env.py --dry-run    # preview changes without writing
    python sync-env.py --regenerate # force-regenerate token variables
"""
import argparse
import configparser
import re
import secrets
import sys
from pathlib import Path

# ── Secret variables (never overwritten after initial write in root .env) ──────

GLOBAL_SECRETS: set[str] = {
    "CF_TUNNEL_TOKEN",
    "CF_ACCESS_HOSTNAME",
    "CF_ACCESS_GOOGLE_CLIENT_ID",
    "CF_ACCESS_GOOGLE_CLIENT_SECRET",
    "HF_TOKEN",
}

# Auto-generated as random tokens on first run (re-generated with --regenerate)
AUTO_GENERATE_TOKENS: frozenset[str] = frozenset({"LLAMA_API_KEY", "MCP_API_KEY"})

# ── Env file parser ────────────────────────────────────────────────────────────

_VAR_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def parse_env(path: Path) -> dict[str, str]:
    """Return {KEY: RAW_VALUE} for every variable in the file."""
    result: dict[str, str] = {}
    if not path.exists():
        return result

    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _VAR_RE.match(line)
        if not m:
            i += 1
            continue

        key, value = m.group(1), m.group(2)

        # Detect multiline quoted strings
        if value and value[0] in ('"', "'"):
            q = value[0]
            stripped = value.replace("\\" + q, "")
            if stripped.count(q) % 2 == 1:   # odd quotes → multiline
                collected = [value]
                i += 1
                while i < len(lines):
                    cont = lines[i]
                    collected.append(cont)
                    i += 1
                    if cont.replace("\\" + q, "").count(q) % 2 == 1:
                        break
                value = "\n".join(collected)

        result[key] = value
        i += 1

    return result


# ── Merge logic (root .env only) ───────────────────────────────────────────────

def merge(
    defaults: dict[str, str],
    current: dict[str, str],
    effective_secrets: set[str],
) -> dict[str, str]:
    """Return only keys present in defaults, preserving secret values.

    Keys in current absent from defaults are omitted — they will be removed
    from the target file on the next write.
    """
    result: dict[str, str] = {}
    for key, value in defaults.items():
        if key not in current:
            result[key] = value                  # new key → add with default
        elif key in effective_secrets:
            result[key] = current[key]           # secret → keep existing value
        else:
            result[key] = value                  # non-secret → update to latest default
    return result


# ── Token generation ───────────────────────────────────────────────────────────

def generate_token() -> str:
    return secrets.token_urlsafe(32)


# ── Writers ───────────────────────────────────────────────────────────────────

def write_env(path: Path, merged: dict[str, str]) -> None:
    """Write merged variables to path, preserving existing key order, appending new keys."""
    out: list[str] = []
    seen: set[str] = set()

    if path.exists():
        raw = path.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(raw):
            line = raw[i]
            m = _VAR_RE.match(line)
            if not m:
                i += 1
                continue

            key, value = m.group(1), m.group(2)
            seen.add(key)

            # Detect multiline so we skip continuation lines
            is_multiline = False
            if value and value[0] in ('"', "'"):
                q = value[0]
                if value.replace("\\" + q, "").count(q) % 2 == 1:
                    is_multiline = True

            if key in merged:
                out.append(f"{key}={merged[key]}")
            # key absent from merged → removed from .env.default; drop line
            i += 1

            if is_multiline:
                i += 1
                while i < len(raw):
                    cont = raw[i]
                    if cont.replace("\\" + value[0], "").count(value[0]) % 2 == 1:
                        break
                    i += 1

    # Append keys that weren't in the existing file
    for key in sorted(merged.keys() - seen):
        out.append(f"{key}={merged[key]}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ── Per-target sync ────────────────────────────────────────────────────────────

def sync_target(
    source_defaults: dict[str, str],
    target: Path,
    effective_secrets: set[str],
    auto_tokens: frozenset[str],
    regenerate: bool,
    dry_run: bool,
    label: str,
) -> bool:
    """Sync source_defaults → target. Returns True if changes were made."""
    current = parse_env(target)

    generated: dict[str, str] = {}
    for key in auto_tokens:
        if key in source_defaults:
            if regenerate or not current.get(key, ""):
                generated[key] = generate_token()

    merged = merge(source_defaults, current, effective_secrets)
    merged.update(generated)

    added     = [k for k in merged if k not in current]
    updated   = [k for k in merged if k in current and k not in effective_secrets and merged[k] != current[k]]
    removed   = [k for k in current if k not in merged]
    regen     = list(generated.keys())
    preserved = [k for k in source_defaults if k in current and k in effective_secrets and source_defaults[k] != current[k]]

    prefix = f"  [{label}]"
    if added:     print(f"{prefix} Add:      {', '.join(sorted(added))}")
    if updated:   print(f"{prefix} Update:   {', '.join(sorted(updated))}")
    if removed:   print(f"{prefix} Remove:   {', '.join(sorted(removed))}")
    if regen:     print(f"{prefix} Generate: {', '.join(sorted(regen))}")
    if preserved: print(f"{prefix} Preserve: {', '.join(sorted(preserved))}")
    if not added and not updated and not removed and not regen:
        print(f"{prefix} No changes needed.")
        return False

    if dry_run:
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    write_env(target, merged)
    print(f"{prefix} ✓ Wrote {target}")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync .env.default → root .env"
    )
    parser.add_argument("--dry-run",      action="store_true", help="Preview changes without writing")
    parser.add_argument("--regenerate",   action="store_true", help="Force-regenerate auto-generated tokens")
    parser.add_argument("--default-file", default=".env.default", help="Path to the root .env.default")
    parser.add_argument("--env-file",     default=".env",         help="Path to the root .env output")
    args = parser.parse_args()

    default_path = Path(args.default_file)
    if not default_path.exists():
        print(f"Error: {default_path} not found.", file=sys.stderr)
        sys.exit(1)

    all_defaults = parse_env(default_path)
    if not all_defaults:
        print(f"Warning: no variables found in {default_path}", file=sys.stderr)

    changed = False

    # ── 1. Root .env (all variables; consumed by docker-compose) ──────────────
    root_env_path = Path(args.env_file)
    root_secrets = GLOBAL_SECRETS | {"MCP_API_KEY", "SEARCH_API_KEY", "LLAMA_API_KEY"}
    changed |= sync_target(
        source_defaults=all_defaults,
        target=root_env_path,
        effective_secrets=root_secrets,
        auto_tokens=AUTO_GENERATE_TOKENS,
        regenerate=args.regenerate,
        dry_run=args.dry_run,
        label=args.env_file,
    )

    if not changed and not args.dry_run:
        print("All env files up to date.")

    # ── 2. opencode.json (substitute {env:VAR} and {ini:...} from models.ini) ──
    sync_opencode(
        env_path=root_env_path,
        template=Path("opencode-default.json"),
        output=Path("opencode.json"),
        dry_run=args.dry_run,
        models_ini_path=Path("config/models.ini"),
    )


def parse_models_ini(path: Path) -> configparser.ConfigParser:
    """Parse models.ini, handling bare key=value lines before the first section."""
    cfg = configparser.ConfigParser(interpolation=None)
    if not path.exists():
        return cfg
    raw = path.read_text(encoding="utf-8")
    # models.ini starts with bare `version = 1` before any section header;
    # configparser requires a section, so we prepend a synthetic [__preamble__].
    cfg.read_string("[__preamble__]\n" + raw)
    return cfg


def sync_opencode(
    env_path: Path,
    template: Path,
    output: Path,
    dry_run: bool,
    models_ini_path: Path | None = None,
) -> None:
    """Substitute {env:VAR} and "{ini:SECTION.KEY}" placeholders in template → output.

    {env:VAR}         → string value from env_path (quotes kept)
    "{ini:SEC.KEY}"   → numeric value from models_ini_path (surrounding quotes stripped,
                        bare integer emitted so JSON stays valid)
    """
    if not template.exists():
        return

    env = parse_env(env_path) if env_path.exists() else {}
    ini = parse_models_ini(models_ini_path) if models_ini_path else configparser.ConfigParser(interpolation=None)
    text = template.read_text(encoding="utf-8")

    missing: list[str] = []

    def env_replacer(m: re.Match) -> str:
        var = m.group(1)
        val = env.get(var, "")
        if val == "":
            missing.append(f"env:{var}")
            return m.group(0)
        return val

    def ini_replacer(m: re.Match) -> str:
        """Replace quoted "{ini:SECTION.KEY}" with bare numeric value."""
        ref = m.group(1)  # e.g. "qwen3.6-27b.ctx-size"
        # Split on last dot to allow dots in section names (e.g. "qwen3.6-27b")
        dot = ref.rfind(".")
        if dot == -1:
            missing.append(f"ini:{ref}")
            return m.group(0)
        section, key = ref[:dot], ref[dot + 1:]
        if ini.has_option(section, key):
            return ini.get(section, key).strip()
        # Fallback to [*] global defaults
        if ini.has_option("*", key):
            return ini.get("*", key).strip()
        missing.append(f"ini:{ref}")
        return m.group(0)

    result = re.sub(r"\{env:([^}]+)\}", env_replacer, text)
    # Match the full quoted placeholder: "{ini:SECTION.KEY}" → bare value
    result = re.sub(r'"\{ini:([^}]+)\}"', ini_replacer, result)

    if dry_run:
        print(f"  [opencode] Would write {output}")
        if missing:
            print(f"  [opencode] Unresolved (missing/empty): {', '.join(missing)}")
        return

    output.write_text(result, encoding="utf-8")
    print(f"  [opencode] ✓ Wrote {output}")
    if missing:
        print(f"  [opencode] Warning — unresolved placeholders: {', '.join(missing)}")


if __name__ == "__main__":
    main()
