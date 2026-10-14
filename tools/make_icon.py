"""Render the app icons into enegmatic/assets/:

    enegmatic.ico   Windows (16-256 px)
    enegmatic.icns  macOS (16-1024 px, drawn on Apple's icon grid)
    enegmatic.png   512 px logo for the README

    python tools/make_icon.py

The icon is drawn by enegmatic.splash.icon_pixmap, so it always matches the app.
Both container formats store PNG images, so no platform tools are needed.
"""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from enegmatic.splash import icon_pixmap, load_brand_font  # noqa: E402

ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
# ICNS PNG element types and their pixel sizes (the @2x variants share the same images)
ICNS_TYPES = (("icp4", 16), ("icp5", 32), ("ic11", 32), ("icp6", 64), ("ic12", 64),
              ("ic07", 128), ("ic08", 256), ("ic13", 256), ("ic09", 512), ("ic14", 512),
              ("ic10", 1024))


def png_bytes(pixmap) -> bytes:
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.WriteOnly)
    pixmap.save(buf, "PNG")
    buf.close()
    return bytes(data)


def make_ico(family: str) -> bytes:
    images = [png_bytes(icon_pixmap(family, s)) for s in ICO_SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for size, blob in zip(ICO_SIZES, images):
        dim = 0 if size >= 256 else size          # 0 means 256 in the ICO format
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset)
        blobs += blob
        offset += len(blob)
    return header + entries + blobs


def make_icns(family: str) -> bytes:
    cache: dict[int, bytes] = {}
    body = b""
    for kind, size in ICNS_TYPES:
        if size not in cache:
            cache[size] = png_bytes(icon_pixmap(family, size, mac=True))
        body += kind.encode("ascii") + struct.pack(">I", 8 + len(cache[size])) + cache[size]
    return b"icns" + struct.pack(">I", 8 + len(body)) + body


def main():
    app = QApplication(sys.argv)  # needed for fonts / painting
    family = load_brand_font()
    app.processEvents()
    assets = ROOT / "enegmatic" / "assets"
    (assets / "enegmatic.ico").write_bytes(make_ico(family))
    (assets / "enegmatic.icns").write_bytes(make_icns(family))
    icon_pixmap(family, 512).save(str(assets / "enegmatic.png"))
    print("wrote enegmatic.ico, enegmatic.icns and enegmatic.png in", assets)


if __name__ == "__main__":
    main()
