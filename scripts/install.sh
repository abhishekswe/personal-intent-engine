#!/bin/bash
# PIE install script — downloads the latest release and installs PIE.app to
# /Applications, stripping the macOS quarantine attribute so Gatekeeper does not
# block the (ad-hoc signed) app on first launch.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/abhishekswe/personal-intent-engine/main/scripts/install.sh | bash
#
# Or, to review first:
#   curl -fsSL https://raw.githubusercontent.com/abhishekswe/personal-intent-engine/main/scripts/install.sh -o install.sh
#   less install.sh && bash install.sh
set -euo pipefail

OWNER="abhishekswe"
REPO="personal-intent-engine"
APP_NAME="PIE"
APP_BUNDLE="${APP_NAME}.app"
INSTALL_DIR="/Applications"
BUNDLE_ID="com.pie.desktop"

# --- helpers -----------------------------------------------------------------
info()  { printf "\033[1;34m==>\033[0m %s\n" "$1"; }
ok()    { printf "\033[1;32m✓\033[0m %s\n" "$1"; }
err()   { printf "\033[1;31m✗\033[0m %s\n" "$1" >&2; }
die()   { err "$1"; exit 1; }

# --- preflight ---------------------------------------------------------------
# macOS only
if [[ "$(uname)" != "Darwin" ]]; then
  die "This installer is for macOS. Windows/Linux builds are on the releases page."
fi

# Apple Silicon only (the released DMG is aarch64)
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  die "PIE releases are currently Apple Silicon (arm64) only. Found: $ARCH.
Build from source on other archs: https://github.com/${OWNER}/${REPO}#quick-start-desktop-app"
fi

# Need write access to /Applications
if [[ ! -w "$INSTALL_DIR" ]]; then
  die "Cannot write to ${INSTALL_DIR}. Run with: sudo env \"PATH=\$PATH\" bash install.sh"
fi

command -v curl    >/dev/null 2>&1 || die "curl is required."
command -v hdiutil >/dev/null 2>&1 || die "hdiutil is required (macOS built-in)."
command -v xattr   >/dev/null 2>&1 || die "xattr is required (macOS built-in)."

# --- find the latest release DMG -------------------------------------------
info "Finding the latest PIE release for Apple Silicon…"
# Use the releases LIST endpoint (includes prereleases). /releases/latest omits
# prereleases, which would 404 while PIE is still in pre-release distribution.
API="https://api.github.com/repos/${OWNER}/${REPO}/releases"
ASSET_JSON=$(curl -fsSL "$API" | python3 -c '
import json, sys
data = json.load(sys.stdin)
if not data:
    sys.exit(0)
r = data[0]  # most recent release, including prereleases
print(r.get("tag_name", ""))
for a in r.get("assets", []):
    n = a.get("name", "")
    if n.endswith("_aarch64.dmg"):
        print(a.get("browser_download_url", ""))
        print(n)
        break
')

VERSION=$(echo "$ASSET_JSON" | sed -n '1p')
DMG_URL=$(echo "$ASSET_JSON" | sed -n '2p')
DMG_NAME=$(echo "$ASSET_JSON" | sed -n '3p')

if [[ -z "$DMG_URL" ]]; then
  die "Could not find an aarch64 .dmg asset in the latest release."
fi
ok "Latest release: ${VERSION}"

# --- download ----------------------------------------------------------------
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"; hdiutil detach "$MOUNT" -quiet 2>/dev/null || true' EXIT

DMG_PATH="${TMPDIR}/${DMG_NAME}"
info "Downloading ${DMG_NAME}…"
curl -fSL "$DMG_URL" -o "$DMG_PATH"
ok "Downloaded to ${DMG_PATH}"

# --- mount -------------------------------------------------------------------
info "Mounting disk image…"
# hdiutil prints a table; the mount point is the last /Volumes/... column.
# Capture full output first so pipefail doesn't abort on grep no-match.
ATTACH_OUT=$(hdiutil attach "$DMG_PATH" -nobrowse || true)
MOUNT=$(echo "$ATTACH_OUT" | grep -oE '/Volumes/.*' | tail -1 | sed 's/[[:space:]]*$//' || true)
[[ -n "$MOUNT" && -d "$MOUNT" ]] || die "Could not mount the disk image."
ok "Mounted at ${MOUNT}"

SRC_APP="${MOUNT}/${APP_BUNDLE}"
[[ -d "$SRC_APP" ]] || die "${APP_BUNDLE} not found inside the disk image."

# --- install -----------------------------------------------------------------
DEST_APP="${INSTALL_DIR}/${APP_BUNDLE}"

# Quit a running instance so we can replace the bundle.
if pgrep -f "${BUNDLE_ID}" >/dev/null 2>&1; then
  info "Quitting a running PIE instance…"
  pkill -f "${BUNDLE_ID}" 2>/dev/null || true
  sleep 1
fi

if [[ -d "$DEST_APP" ]]; then
  info "Replacing existing ${DEST_APP}…"
  rm -rf "$DEST_APP"
fi

info "Copying ${APP_BUNDLE} to ${INSTALL_DIR}…"
cp -R "$SRC_APP" "$DEST_APP"
ok "Installed ${DEST_APP}"

# --- strip quarantine so Gatekeeper does not block the ad-hoc signed app -----
info "Removing Gatekeeper quarantine attribute…"
xattr -cr "$DEST_APP"
if xattr -l "$DEST_APP" 2>/dev/null | grep -q "com.apple.quarantine"; then
  err "Quarantine attribute could not be removed. First launch may be blocked."
  err "Fix manually:  xattr -cr \"${DEST_APP}\""
else
  ok "Quarantine removed — PIE will open without a Gatekeeper prompt."
fi

# --- done --------------------------------------------------------------------
printf '\033[1;32m%s\033[0m\n' "🍺  PIE ${VERSION} installed!"
printf '\n%s\n' "Next steps:"
printf '  1. %s\n' "Launch PIE from Applications (or Spotlight):  open -a PIE"
printf '  2. %s\n' "On first run, macOS asks for Microphone and Accessibility permission."
printf '     %s\n' "Grant both — Accessibility is needed to paste the prompt at your cursor."
printf '  3. %s\n' "Open the Models tab and download a whisper model (start with Whisper Tiny)"
printf '     %s\n' "and Silero VAD."
printf '  4. %s\n' "Press ⌘⇧Space (rebindable) in any app to start recording."
printf '\nDocs:  https://github.com/%s/%s\n' "${OWNER}" "${REPO}"
printf 'Issues: https://github.com/%s/%s/issues\n\n' "${OWNER}" "${REPO}"