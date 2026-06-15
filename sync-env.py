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

opencode.json is generated from opencode-default.json by substituting:
  {env:VAR}    → value from root .env
  {ini:SEC.KEY} → value from config/models.ini
  "{ctx:MODEL}" → model context length (ctx-size from models.ini, else fit-ctx
                  floor, else 200000), emitted as bare integer

The provider "models" block in opencode.json comes from the static template,
with {ctx:...} substitution from models.ini.

Usage:
    python sync-env.py                              # sync root .env
    python sync-env.py --dry-run                    # preview changes without writing
    python sync-env.py --regenerate                 # force-regenerate token variables
"""
import argparse
import configparser
import json as _json
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
    "CF_ACCESS_CLIENT_ID",
    "CF_ACCESS_CLIENT_SECRET",
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


# ── Context length resolution ──────────────────────────────────────────────────

def resolve_context_lengths(
    ini: configparser.ConfigParser,
    model_names: list[str],
) -> dict[str, int]:
    """Return {model: context_length} for each model name.

    Priority:
      1. ctx-size from model's INI section (if set explicitly)
      2. fit-ctx from [*] global defaults (the minimum fit floor)
      3. 200000 as hard fallback
    """
    # fit-ctx from [*] (may be commented out; fallback to 200000)
    fit_ctx_floor = 200_000
    if ini.has_option("*", "fit-ctx"):
        try:
            fit_ctx_floor = int(ini.get("*", "fit-ctx").strip())
        except ValueError:
            pass

    results: dict[str, int] = {}
    for name in model_names:
        # Try explicit ctx-size in the model section
        if ini.has_option(name, "ctx-size"):
            try:
                results[name] = int(ini.get(name, "ctx-size").strip())
                print(f"  [ctx]     '{name}' ctx-size = {results[name]:,} (from INI)")
                continue
            except ValueError:
                pass

        # Fall back to fit-ctx floor
        results[name] = fit_ctx_floor
        print(f"  [ctx]     '{name}' ctx-size not set — using fit-ctx floor: {fit_ctx_floor:,}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync .env.default → root .env"
    )
    parser.add_argument("--dry-run",      action="store_true", help="Preview changes without writing")
    parser.add_argument("--regenerate",   action="store_true", help="Force-regenerate auto-generated tokens")
    parser.add_argument("--default-file", default=".env.default", help="Path to the root .env.default")
    parser.add_argument("--env-file",     default=".env",         help="Path to the root .env output")
    parser.add_argument("--clear-kv-cache", action="store_true", help="Delete all saved KV cache slot files from models/slots/")
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

    # ── 1b. webui-config.json — populate API keys from .env ─────────────
    sync_webui_config(
        default_path=Path("config/webui-config.default.json"),
        output_path=Path("config/webui-config.json"),
        env_path=root_env_path,
        dry_run=args.dry_run,
    )

    # ── 3. Resolve context lengths (from models.ini) ──────────────────────────
    ini = parse_models_ini(Path("config/models.ini"))

    # ── 2b. Slot save-path directories ────────────────────────────────────────
    sync_slot_dirs(ini, dry_run=args.dry_run)

    # ── 2c. Clear saved KV cache slot files ───────────────────────────────────
    if args.clear_kv_cache:
        clear_kv_cache(ini, dry_run=args.dry_run)

    # Collect model names from non-global, non-preamble sections
    model_names = [
        s for s in ini.sections()
        if s not in ("__preamble__", "*")
    ]

    ctx_lengths = resolve_context_lengths(ini, model_names)

    # ── 4. opencode.json ──────────────────────────────────────────────────────
    sync_opencode(
        env_path=root_env_path,
        template=Path("opencode-default.json"),
        output=Path("opencode.json"),
        dry_run=args.dry_run,
        models_ini_path=Path("config/models.ini"),
        ctx_lengths=ctx_lengths,
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
    ctx_lengths: dict[str, int] | None = None,
) -> None:
    """Substitute placeholders in template → output.

    {env:VAR}         → string value from env_path (quotes kept)
    "{ini:SEC.KEY}"   → numeric value from models_ini_path (surrounding quotes stripped,
                        bare integer emitted so JSON stays valid)
    "{ctx:MODEL}"     → resolved context length for MODEL (bare integer, no quotes)

    The provider "models" block is taken from the static template, with
    {ctx:...} substitution from models.ini.
    """
    if not template.exists():
        return

    env = parse_env(env_path) if env_path.exists() else {}
    ini = parse_models_ini(models_ini_path) if models_ini_path else configparser.ConfigParser(interpolation=None)
    ctx = ctx_lengths or {}
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
        ref = m.group(1)  # e.g. "qwopus3.6-27b.ctx-size"
        dot = ref.rfind(".")
        if dot == -1:
            missing.append(f"ini:{ref}")
            return m.group(0)
        section, key = ref[:dot], ref[dot + 1:]
        if ini.has_option(section, key):
            return ini.get(section, key).strip()
        if ini.has_option("*", key):
            return ini.get("*", key).strip()
        missing.append(f"ini:{ref}")
        return m.group(0)

    def ctx_replacer(m: re.Match) -> str:
        """Replace quoted "{ctx:MODEL}" with bare integer context length."""
        model = m.group(1)
        if model in ctx:
            return str(ctx[model])
        missing.append(f"ctx:{model}")
        return m.group(0)

    result = re.sub(r"\{env:([^}]+)\}", env_replacer, text)
    result = re.sub(r'"\{ini:([^}]+)\}"', ini_replacer, result)
    result = re.sub(r'"\{ctx:([^}]+)\}"', ctx_replacer, result)

    if dry_run:
        print(f"  [opencode] Would write {output}")
        if missing:
            print(f"  [opencode] Unresolved (missing/empty): {', '.join(missing)}")
        return

    output.write_text(result, encoding="utf-8")
    print(f"  [opencode] ✓ Wrote {output}")
    if missing:
        print(f"  [opencode] Warning — unresolved placeholders: {', '.join(missing)}")


def clear_kv_cache(ini: configparser.ConfigParser, dry_run: bool) -> None:
    """Delete all saved KV cache files from every slot-save-path in models.ini.

    Deletes files inside each slot dir but preserves the directories themselves
    (sync_slot_dirs manages their lifecycle).  Skips dirs that don't exist.
    """
    CONTAINER_PREFIX = "/models/"
    HOST_PREFIX = "models/"

    found_any = False
    for section in ini.sections():
        if section in ("__preamble__", "*"):
            continue
        raw = ini.get(section, "slot-save-path", fallback="").strip()
        if not raw or not raw.startswith(CONTAINER_PREFIX):
            continue
        host_path = Path(HOST_PREFIX + raw[len(CONTAINER_PREFIX):])
        if not host_path.exists():
            continue

        files = sorted(host_path.iterdir()) if host_path.is_dir() else []
        if not files:
            continue

        found_any = True
        for f in files:
            if not f.is_file():
                continue
            if dry_run:
                print(f"  [kv-cache] Would delete: {f}")
            else:
                f.unlink()
                print(f"  [kv-cache] ✗ Deleted:    {f}")

    if not found_any:
        print("  [kv-cache] No saved KV cache files found.")


def sync_slot_dirs(ini: configparser.ConfigParser, dry_run: bool) -> None:
    """Create slot-save-path dirs for all models in models.ini; remove unused ones.

    Paths in models.ini use the container namespace (/models/...).
    The host-side mount is ./models (docker-compose: ./models:/models),
    so /models/slots/foo → ./models/slots/foo on the host.

    Creates:  any slot dir referenced by slot-save-path that does not exist.
    Removes:  any subdirectory of ./models/slots/ NOT referenced by any model.
              Skipped entirely if the expected set is empty (parse miss guard).
    """
    CONTAINER_PREFIX = "/models/"
    HOST_PREFIX = "models/"

    def container_to_host(container_path: str) -> Path | None:
        """Translate /models/slots/foo → Path('models/slots/foo')."""
        if container_path.startswith(CONTAINER_PREFIX):
            return Path(HOST_PREFIX + container_path[len(CONTAINER_PREFIX):])
        return None

    # Collect all slot-save-path values from non-global sections
    expected: dict[str, Path] = {}  # section → host path
    for section in ini.sections():
        if section in ("__preamble__", "*"):
            continue
        raw = ini.get(section, "slot-save-path", fallback="").strip()
        if not raw:
            continue
        host_path = container_to_host(raw)
        if host_path is None:
            print(f"  [slots]   Warning: unrecognised slot-save-path for [{section}]: {raw!r}")
            continue
        expected[section] = host_path

    # Parse guard: if we found no slot dirs, don't touch anything
    if not expected:
        return

    slots_root = Path("models/slots")

    # ── Create missing dirs ────────────────────────────────────────────────────
    for section, host_path in sorted(expected.items()):
        if host_path.exists():
            continue
        if dry_run:
            print(f"  [slots]   Would create: {host_path}")
        else:
            host_path.mkdir(parents=True, exist_ok=True)
            print(f"  [slots]   ✓ Created:    {host_path}")

    # ── Remove unused subdirs ─────────────────────────────────────────────────
    if not slots_root.exists():
        return

    expected_names = {p.resolve() for p in expected.values()}

    for candidate in sorted(slots_root.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.resolve() in expected_names:
            continue
        if dry_run:
            print(f"  [slots]   Would remove: {candidate} (not referenced by any model)")
        else:
            import shutil
            shutil.rmtree(candidate)
            print(f"  [slots]   ✗ Removed:    {candidate} (not referenced by any model)")


def sync_webui_config(
    default_path: Path,
    output_path: Path,
    env_path: Path,
    dry_run: bool,
) -> bool:
    """Generate webui-config.json from webui-config.default.json, populating API keys.

    - Sets top-level apiKey from LLAMA_API_KEY.
    - Fills the Authorization header inside mcpServers from MCP_API_KEY.

    Returns True if output would change (or did change).
    """
    if not default_path.exists():
        return False

    env = parse_env(env_path) if env_path.exists() else {}

    llama_key = env.get("LLAMA_API_KEY", "")
    mcp_key = env.get("MCP_API_KEY", "")

    try:
        config = _json.loads(default_path.read_text(encoding="utf-8"))
    except (_json.JSONDecodeError, OSError) as e:
        print(f"  [webui] Warning — could not read {default_path}: {e}")
        return False

    # ── Top-level apiKey ─────────────────────────────────────────────
    if llama_key:
        config["apiKey"] = llama_key

    # ── mcpServers Authorization header ──────────────────────────────
    mcp_str = config.get("mcpServers", "")
    if isinstance(mcp_str, str) and mcp_str:
        try:
            servers = _json.loads(mcp_str)
        except _json.JSONDecodeError:
            pass
        else:
            if isinstance(servers, list) and mcp_key:
                for srv in servers:
                    if not isinstance(srv, dict):
                        continue
                    hdrs_raw = srv.get("headers", "")
                    if not isinstance(hdrs_raw, str) or not hdrs_raw:
                        continue
                    try:
                        hdrs = _json.loads(hdrs_raw)
                    except _json.JSONDecodeError:
                        continue
                    if not isinstance(hdrs, dict):
                        continue
                    hdrs["Authorization"] = f"Bearer {mcp_key}"
                    srv["headers"] = _json.dumps(hdrs)
                config["mcpServers"] = _json.dumps(servers)

    new_text = _json.dumps(config, indent=2, ensure_ascii=False) + "\n"

    # Compare with existing output
    if output_path.exists():
        try:
            old_text = output_path.read_text(encoding="utf-8")
        except OSError:
            old_text = None
        if old_text == new_text:
            return False

    if dry_run:
        if llama_key:
            print(f"  [webui] apiKey: {llama_key[:6]}…")
        if mcp_key:
            print(f"  [webui] mcpServers auth: Bearer {mcp_key[:6]}…")
        print(f"  [webui] Would write {output_path}")
        return True

    output_path.write_text(new_text, encoding="utf-8")
    print(f"  [webui] ✓ Generated {output_path}")
    return True


if __name__ == "__main__":
    main()
