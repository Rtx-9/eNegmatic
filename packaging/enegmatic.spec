# PyInstaller build for eNegmatic (Windows, one-folder app).
#
#     build.bat                      (or)
#     python -m PyInstaller --noconfirm --clean packaging/enegmatic.spec
#
# Output: dist/eNegmatic/eNegmatic.exe plus its support files.
import os

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(ROOT, "enegmatic", "assets", "enegmatic.ico")

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=collect_dynamic_libs("rawpy"),          # LibRaw
    datas=[
        (os.path.join(ROOT, "enegmatic", "assets"), "enegmatic/assets"),
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
    icon=ICON,
    version=os.path.join(SPECPATH, "version_info.txt"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="eNegmatic", upx=False)
