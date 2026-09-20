#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALLER="${SCRIPT_DIR}/install.sh"

if grep -q 'hdiutil' "$INSTALLER"; then
  echo "installer still uses deprecated hdiutil" >&2
  exit 1
fi

grep -q 'diskutil image attach --nobrowse' "$INSTALLER"
grep -q 'diskutil eject' "$INSTALLER"
echo "install script checks passed"
