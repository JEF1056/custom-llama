#!/usr/bin/env bash
#
# OpenCode + Olla Sticky Session Installer
# Installs opencode.json and persistent session affinity plugins into any repository.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$SCRIPT_DIR/templates"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

show_help() {
    cat << 'EOF'
Usage:
  ./install.sh [TARGET_DIR] [OPTIONS]

Arguments:
  TARGET_DIR       Directory to install into (default: current working directory)

Options:
  -g, --global     Install globally into ~/.config/opencode
  -f, --force      Overwrite existing opencode.json without backup prompt
  -h, --help       Show this help message

Examples:
  ./install.sh                     # Install into current repo
  ./install.sh /path/to/my-repo    # Install into specific repo
  ./install.sh --global            # Install globally into user config
EOF
}

TARGET_DIR=""
GLOBAL_MODE=0
FORCE_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--global)
            GLOBAL_MODE=1
            shift
            ;;
        -f|--force)
            FORCE_MODE=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$TARGET_DIR" ]]; then
                TARGET_DIR="$1"
            else
                echo -e "${RED}Error: Unknown argument: $1${NC}" >&2
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ $GLOBAL_MODE -eq 1 ]]; then
    TARGET_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
elif [[ -z "$TARGET_DIR" ]]; then
    TARGET_DIR="$(pwd)"
fi

# Resolve absolute path
TARGET_DIR="$(cd "$TARGET_DIR" 2>/dev/null && pwd || echo "$TARGET_DIR")"

echo -e "${BLUE}=== OpenCode + Olla Installer ===${NC}"
echo -e "Target Directory: ${GREEN}$TARGET_DIR${NC}"

# Ensure target directory exists
mkdir -p "$TARGET_DIR"

# 1. Install plugins (.opencode/plugins and .opencode/plugin)
PLUGIN_DIR="$TARGET_DIR/.opencode/plugins"
LEGACY_PLUGIN_DIR="$TARGET_DIR/.opencode/plugin"
mkdir -p "$PLUGIN_DIR" "$LEGACY_PLUGIN_DIR"

echo -e "Installing session affinity plugins..."
cp "$TEMPLATES_DIR/.opencode/plugins/olla-session.js" "$PLUGIN_DIR/olla-session.js"
cp "$TEMPLATES_DIR/.opencode/plugin/sticky-header.js" "$LEGACY_PLUGIN_DIR/sticky-header.js"
echo -e "  ${GREEN}✓${NC} $PLUGIN_DIR/olla-session.js"
echo -e "  ${GREEN}✓${NC} $LEGACY_PLUGIN_DIR/sticky-header.js"

# 2. Install opencode.json
CONFIG_DEST="$TARGET_DIR/opencode.json"
if [[ -f "$CONFIG_DEST" && $FORCE_MODE -eq 0 ]]; then
    BACKUP_DEST="$CONFIG_DEST.bak.$(date +%s)"
    echo -e "${YELLOW}Warning: $CONFIG_DEST already exists.${NC}"
    echo -e "Creating backup -> ${YELLOW}$BACKUP_DEST${NC}"
    cp "$CONFIG_DEST" "$BACKUP_DEST"
fi

cp "$TEMPLATES_DIR/opencode.json" "$CONFIG_DEST"
echo -e "  ${GREEN}✓${NC} $CONFIG_DEST"

echo -e "${GREEN}=== Installation Complete! ===${NC}"
echo -e "Provider configured: ${BLUE}olla${NC} (model: ${BLUE}olla/qwen3.8-27b${NC})"
echo -e "Persistent session header: ${BLUE}X-Olla-Session-ID${NC} enabled for KV-cache reuse."
