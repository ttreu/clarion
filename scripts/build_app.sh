#!/bin/bash
# Builds clarion.app and a zip for GitHub Releases (dist/clarion-<version>-macos-<arch>.zip).
# The app is ad-hoc signed (runs on this architecture) but not notarized by Apple: on first launch
# macOS asks for confirmation (README → "Install and run").
# The build runs outside the project folder: iCloud adds file attributes there that codesign rejects.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
uv sync --extra build
VERSION=$(uv run python -c "import klangrestaurator as k; print(k.__version__)")
ARCH=$(uname -m)
WORK="${TMPDIR:-/tmp}/clarion-build"
rm -rf "$WORK"
mkdir -p "$WORK"
uv run pyinstaller --noconfirm --windowed \
  --name clarion \
  --osx-bundle-identifier com.clarion.app \
  --icon "$ROOT/assets/icon/clairon.icns" \
  --add-data "$ROOT/src/klangrestaurator/gui/assets:klangrestaurator/gui/assets" \
  --distpath "$WORK/dist" --workpath "$WORK/build" --specpath "$WORK" \
  --collect-submodules noisereduce \
  --collect-submodules nara_wpe \
  --add-data "$ROOT/LICENSE:." \
  --add-data "$ROOT/THIRD_PARTY_NOTICES.md:." \
  --paths "$ROOT/src" \
  "$ROOT/packaging/launch.py"
APP="$WORK/dist/clarion.app"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$APP/Contents/Info.plist"
xattr -cr "$APP"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
mkdir -p dist
ZIP="dist/clarion-$VERSION-macos-$ARCH.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
echo "App:  $APP"
echo "Zip:  $ROOT/$ZIP  (upload this to a GitHub release)"
