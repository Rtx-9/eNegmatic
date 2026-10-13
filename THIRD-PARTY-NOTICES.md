# Third-party software

eNegmatic is licensed under the GNU General Public License v3.0 (see `LICENSE`).
The Windows build bundles the following components, each under its own licence.

| Component | Used for | Licence | Source |
|---|---|---|---|
| Python | runtime | PSF License | https://www.python.org |
| Qt / PySide6 (Qt for Python) | user interface | LGPL-3.0 | https://code.qt.io |
| rawpy | raw decoding (Python bindings) | MIT | https://github.com/letmaik/rawpy |
| LibRaw (bundled with rawpy) | raw decoding | LGPL-2.1 or CDDL-1.0 | https://www.libraw.org |
| OpenCV (opencv-python-headless) | image processing | Apache-2.0 | https://github.com/opencv/opencv-python |
| NumPy | numerical processing | BSD-3-Clause | https://numpy.org |
| PyInstaller bootloader (Windows build only) | packaging | GPL-2.0 with bootloader exception | https://pyinstaller.org |
| Comfortaa font (`enegmatic/assets/Comfortaa.ttf`) | splash screen / icon fallback | SIL Open Font License 1.1 (`enegmatic/assets/OFL.txt`) | https://github.com/googlefonts/comfortaa |

The LGPL-licensed libraries (Qt, LibRaw) are dynamically linked and can be replaced
with modified versions by swapping the corresponding DLLs in the app folder.

Century Gothic, used for the splash screen when it is installed, is **not** included;
it is loaded from the user's system.
