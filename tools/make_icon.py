"""Render the app icon to enegmatic/assets/enegmatic.ico (+ a 512 px PNG logo).

    python tools/make_icon.py

The icon is drawn by enegmatic.splash.icon_pixmap, so it always matches the app.
The .ico stores PNG-compressed images, which Windows Vista and later read.
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from enegmatic.splash import icon_pixmap, load_brand_font  # noqa: E402

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def png_bytes(pixmap) -> bytes:
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    buf.close()
    return bytes(data)


def main():
    app = QApplication(sys.argv)  # needed for fonts / painting
    family = load_brand_font()
    app.processEvents()
    images = [png_bytes(icon_pixmap(family, s)) for s in SIZES]

    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, blob in zip(SIZES, images):
        dim = 0 if size >= 256 else size          # 0 means 256 in the ICO format
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset)
        blobs += blob
        offset += len(blob)

    assets = ROOT / "enegmatic" / "assets"
    (assets / "enegmatic.ico").write_bytes(header + entries + blobs)
    icon_pixmap(family, 512).save(str(assets / "enegmatic.png"))
    print("wrote", assets / "enegmatic.ico", "and", assets / "enegmatic.png")


if __name__ == "__main__":
    main()
