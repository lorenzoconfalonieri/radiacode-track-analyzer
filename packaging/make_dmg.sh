#!/bin/sh
# Wrap the PyInstaller .app in a compressed disk image with the usual
# drag-to-Applications link.  Usage: packaging/make_dmg.sh OUTPUT.dmg
set -eu
APP="dist/Radiacode Track Analyzer.app"
OUT="$1"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$OUT"
hdiutil create -volname "Radiacode Track Analyzer" -srcfolder "$STAGE" \
    -fs HFS+ -format UDZO -imagekey zlib-level=9 "$OUT"
