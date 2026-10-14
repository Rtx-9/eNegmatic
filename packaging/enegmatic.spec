# PyInstaller build for eNegmatic.
#
#     Windows:  build.bat      ->  dist/eNegmatic/eNegmatic.exe
#     macOS:    ./build.sh     ->  dist/eNegmatic.app
#
# (or: python -m PyInstaller --noconfirm --clean packaging/enegmatic.spec)
# PyInstaller cannot cross-compile: build each platform on that platform.
import os
import re
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ASSETS = os.path.join(ROOT, "enegmatic", "assets")
MAC = sys.platform == "darwin"
with open(os.path.join(ROOT, "enegmatic", "__init__.py"), encoding="utf-8") as f:
    VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', f.read()).group(1)

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=collect_dynamic_libs("rawpy"),          # LibRaw
    datas=[
        (ASSETS, "enegmatic/assets"),
        (os.path.join(ROOT, "LICENSE"), "."),
        (os.path.join(ROOT, "THIRD-PARTY-NOTICES.md"), "."),
    ],
    hiddenimports=[],
    excludes=["tkinter", "matplotlib", "IPython", "PySide6.QtWebEngineCore",
              "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="eNegmatic",
    icon=os.path.join(ASSETS, "enegmatic.icns" if MAC else "enegmatic.ico"),
    version=None if MAC else os.path.join(SPECPATH, "version_info.txt"),
    console=False,
    upx=False,
    argv_emulation=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="eNegmatic", upx=False)

if MAC:
    app = BUNDLE(
        coll,
        name="eNegmatic.app",
        icon=os.path.join(ASSETS, "enegmatic.icns"),
        bundle_identifier="io.github.rtx-9.enegmatic",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": "eNegmatic",
            "CFBundleName": "eNegmatic",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHumanReadableCopyright": "Copyright (C) Rtx-9. Licensed under the GNU GPL v3.0.",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,     # follow macOS dark mode
            "LSMinimumSystemVersion": "12.0",
            "LSApplicationCategoryType": "public.app-category.photography",
        },
    )
