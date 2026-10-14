#!/usr/bin/env bash
# Builds dist/eNegmatic.app and dist/eNegmatic-macos.dmg (run on a Mac).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt pyinstaller
.venv/bin/python -m PyInstaller --noconfirm --clean packaging/enegmatic.spec

# Disk image with the app and an Applications shortcut (drag to install)
rm -rf dist/dmg dist/eNegmatic-macos.dmg
mkdir -p dist/dmg
cp -R dist/eNegmatic.app dist/dmg/
ln -s /Applications dist/dmg/Applications
hdiutil create -volname eNegmatic -srcfolder dist/dmg -ov -format UDZO dist/eNegmatic-macos.dmg
rm -rf dist/dmg

echo
echo "Done: dist/eNegmatic.app  and  dist/eNegmatic-macos.dmg"
