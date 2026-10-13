"""Loading negatives (raw + bitmap), saving positives, and per-folder sidecar files."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .core import resize_long_edge

RAW_EXTS = {".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2", ".raf",
            ".orf", ".rw2", ".pef", ".dng", ".srw", ".3fr", ".iiq", ".erf", ".kdc",
            ".mos", ".mrw", ".rwl", ".x3f"}
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SUPPORTED_EXTS = RAW_EXTS | IMAGE_EXTS

SIDECAR_NAME = ".enegmatic.json"
LEGACY_SIDECARS = (".filmreverse.json", ".negamatic.json")   # read so edits from earlier versions survive
PROXY_EDGE = 1600
THUMB_EDGE = 240

FORMATS = {
    "jpeg": (".jpg", "JPEG"),
    "tiff": (".tif", "TIFF"),
    "png": (".png", "PNG"),
}


def is_supported(path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTS


def collect_files(paths) -> list[str]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(str(c) for c in sorted(p.iterdir()) if c.is_file() and is_supported(c))
        elif p.is_file() and is_supported(p):
            out.append(str(p))
    seen, unique = set(), []
    for p in out:
        key = str(Path(p).resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def load_negative(path: str, half_size: bool = False) -> np.ndarray:
    """Load a negative as linear-ish float32 RGB in [0, 1]."""
    if Path(path).suffix.lower() in RAW_EXTS:
        out = _load_raw_u16(path, half_size).astype(np.float32)
        out *= 1.0 / 65535.0
        return out
    return _load_bitmap(path)


def load_negative_u16(path: str) -> np.ndarray:
    """Full-resolution negative as uint16 RGB (half the memory of float32)."""
    if Path(path).suffix.lower() in RAW_EXTS:
        return _load_raw_u16(path, False)
    return (np.clip(_load_bitmap(path), 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def _load_raw_u16(path: str, half_size: bool) -> np.ndarray:
    import rawpy

    # Open through a file object: LibRaw can't open non-ASCII paths on Windows.
    with open(path, "rb") as f, rawpy.imread(f) as raw:
        kw = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16,
                  half_size=half_size, output_color=rawpy.ColorSpace.sRGB)
        # A fixed daylight balance keeps frames consistent regardless of camera AWB.
        try:
            wb = [float(v) for v in raw.daylight_whitebalance]
        except Exception:
            wb = []
        if len(wb) >= 3 and min(wb[:3]) > 0:
            if len(wb) < 4 or wb[3] <= 0:
                wb = wb[:3] + [wb[1]]
            kw["user_wb"] = wb[:4]
        else:
            kw["use_camera_wb"] = True
        return raw.postprocess(**kw)


def _load_bitmap(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unsupported or corrupt image: {Path(path).name}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def load_proxy(path: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Preview-sized negative, thumbnail, and the (h, w) of the full-resolution image."""
    is_raw = Path(path).suffix.lower() in RAW_EXTS
    neg = load_negative(path, half_size=is_raw)
    full_shape = (neg.shape[0] * 2, neg.shape[1] * 2) if is_raw else neg.shape[:2]
    proxy = np.ascontiguousarray(resize_long_edge(neg, PROXY_EDGE))
    thumb = np.ascontiguousarray(resize_long_edge(proxy, THUMB_EDGE))
    return proxy, thumb, full_shape


def save_image(path: str, img: np.ndarray, fmt: str, bits: int = 8, quality: int = 95,
               grayscale: bool = False) -> None:
    """Save a float RGB image in [0, 1] (unicode-safe)."""
    if fmt == "jpeg":
        bits = 8
    if bits == 16:
        arr = (img * 65535.0 + 0.5).astype(np.uint16)
    else:
        arr = (img * 255.0 + 0.5).astype(np.uint8)
    arr = np.ascontiguousarray(arr[..., 0] if grayscale else arr[..., ::-1])
    ext = FORMATS[fmt][0]
    if fmt == "jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 5]  # LZW
    ok, buf = cv2.imencode(ext, arr, params)
    if not ok:
        raise IOError(f"Could not encode {Path(path).name}")
    buf.tofile(path)


def _read_sidecar(p: Path) -> dict:
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("images", {}) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_sidecar(folder: str) -> dict:
    images = {}
    for name in LEGACY_SIDECARS + (SIDECAR_NAME,):
        images.update(_read_sidecar(Path(folder) / name))
    return images


def save_sidecar(folder: str, images: dict) -> None:
    p = Path(folder) / SIDECAR_NAME
    merged = load_sidecar(folder)
    merged.update(images)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "images": merged}, f, indent=1)
    tmp.replace(p)
