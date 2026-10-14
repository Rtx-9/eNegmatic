"""Output colour spaces: conversion from the app's sRGB rendering, ICC profiles, embedding.

The pipeline renders display-referred sRGB (what the preview shows). An export in another
colour space re-expresses those same colours: decode sRGB -> linear -> XYZ -> the target's
primaries (Bradford-adapted when the white point differs) -> the target's tone curve.
Every exported file gets a matching ICC profile, generated here from the space's published
primaries, white point and tone curve, so no third-party profile files are needed.
"""
from __future__ import annotations

import datetime
import struct
import zlib
from dataclasses import dataclass

import numpy as np

D65 = (0.3127, 0.3290)
D50 = (0.3457, 0.3585)
D50_XYZ = np.array([0.9642, 1.0, 0.8249])


@dataclass(frozen=True)
class Space:
    key: str
    label: str
    description: str          # ICC profile description
    primaries: tuple          # (xy red, xy green, xy blue)
    white: tuple              # xy
    gamma: float | None       # None = sRGB piecewise curve
    hint: str = ""


SPACES = {
    "srgb": Space("srgb", "sRGB (standard)", "sRGB (eNegmatic)",
                  ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), D65, None,
                  "Best for sharing, the web and most screens."),
    "p3": Space("p3", "Display P3", "Display P3 (eNegmatic)",
                ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), D65, None,
                "Wide-gamut screens: Apple devices, many newer monitors and phones."),
    "adobe": Space("adobe", "Adobe RGB (compatible)", "Adobe RGB (1998) compatible (eNegmatic)",
                   ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)), D65, 563 / 256,
                   "Common for printing and print labs."),
    "prophoto": Space("prophoto", "ProPhoto RGB", "ProPhoto RGB (eNegmatic)",
                      ((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)), D50, 461 / 256,
                      "Very wide gamut for further editing. Use 16-bit TIFF / PNG."),
}
DEFAULT = "srgb"


# --------------------------------------------------------------------------- maths

def _xy_to_xyz(xy) -> np.ndarray:
    x, y = xy
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz(space: Space) -> np.ndarray:
    """Linear RGB -> XYZ (relative to the space's own white, Y of white = 1)."""
    p = np.stack([_xy_to_xyz(c) for c in space.primaries], axis=1)
    s = np.linalg.solve(p, _xy_to_xyz(space.white))
    return p * s


_BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                      [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]])


def adapt(src_white_xyz, dst_white_xyz) -> np.ndarray:
    """Bradford chromatic adaptation matrix (XYZ under src white -> XYZ under dst white)."""
    s, d = _BRADFORD @ src_white_xyz, _BRADFORD @ dst_white_xyz
    return np.linalg.inv(_BRADFORD) @ np.diag(d / s) @ _BRADFORD


def srgb_decode(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def encode(lin: np.ndarray, space: Space) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    if space.gamma is None:
        return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return lin ** (1.0 / space.gamma)


def conversion_matrix(space: Space) -> np.ndarray:
    """Linear sRGB -> linear target RGB."""
    src = SPACES["srgb"]
    m = rgb_to_xyz(src)
    if space.white != src.white:
        m = adapt(_xy_to_xyz(src.white), _xy_to_xyz(space.white)) @ m
    return np.linalg.inv(rgb_to_xyz(space)) @ m


def convert(img: np.ndarray, key: str, grayscale: bool = False) -> np.ndarray:
    """sRGB-encoded float image (H, W, 3) in [0, 1] -> the target space's encoded values.

    Grayscale output (B&W films) only changes the tone curve.
    """
    space = SPACES[key]
    if key == "srgb":
        return img
    out = np.empty_like(img)
    m = conversion_matrix(space).astype(np.float32)
    for y in range(0, img.shape[0], 512):              # in strips to limit memory
        lin = srgb_decode(img[y:y + 512]).astype(np.float32)
        if not grayscale:
            lin = lin @ m.T
        out[y:y + 512] = encode(lin, space)
    return out


# --------------------------------------------------------------------------- ICC profiles

def _s15(v: float) -> bytes:
    return struct.pack(">i", int(round(v * 65536)))


def _xyz_tag(xyz) -> bytes:
    return b"XYZ " + b"\0" * 4 + b"".join(_s15(c) for c in xyz)


def _text_tag(text: str) -> bytes:
    return b"text" + b"\0" * 4 + text.encode("ascii") + b"\0"


def _desc_tag(text: str) -> bytes:
    a = text.encode("ascii") + b"\0"
    return (b"desc" + b"\0" * 4 + struct.pack(">I", len(a)) + a
            + struct.pack(">II", 0, 0) + struct.pack(">HB", 0, 0) + b"\0" * 67)


def _curve_tag(space: Space) -> bytes:
    if space.gamma is not None:
        return b"curv" + b"\0" * 4 + struct.pack(">IH", 1, int(round(space.gamma * 256)))
    x = np.linspace(0.0, 1.0, 1024)
    table = np.round(srgb_decode(x) * 65535).astype(">u2").tobytes()
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1024) + table


def _pad4(b: bytes) -> bytes:
    return b + b"\0" * (-len(b) % 4)


def icc_profile(key: str, grayscale: bool = False) -> bytes:
    """ICC v2 display profile (RGB matrix/TRC, or gray TRC) for an output space."""
    space = SPACES[key]
    white = _xy_to_xyz(space.white)
    tags = [(b"desc", _desc_tag(space.description + (" gray" if grayscale else ""))),
            (b"cprt", _text_tag("No copyright, use freely. Generated by eNegmatic.")),
            (b"wtpt", _xyz_tag(white))]
    curve = _curve_tag(space)
    if grayscale:
        tags.append((b"kTRC", curve))
    else:
        # colorants adapted to the D50 profile connection space (Bradford)
        m = adapt(white, D50_XYZ) @ rgb_to_xyz(space)
        tags += [(b"rXYZ", _xyz_tag(m[:, 0])), (b"gXYZ", _xyz_tag(m[:, 1])),
                 (b"bXYZ", _xyz_tag(m[:, 2])),
                 (b"rTRC", curve), (b"gTRC", curve), (b"bTRC", curve)]

    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    entries, data, placed = b"", b"", {}
    for sig, body in tags:
        if body in placed:                       # identical curves share one copy
            entries += struct.pack(">4sII", sig, placed[body], len(body))
            continue
        placed[body] = offset + len(data)
        entries += struct.pack(">4sII", sig, placed[body], len(body))
        data += _pad4(body)
    total = 128 + table_size + len(data)
    now = datetime.datetime.now(datetime.timezone.utc)
    header = (struct.pack(">I", total) + b"\0" * 4 + struct.pack(">I", 0x02100000)
              + b"mntr" + (b"GRAY" if grayscale else b"RGB ") + b"XYZ "
              + struct.pack(">6H", now.year, now.month, now.day, now.hour, now.minute, now.second)
              + b"acsp" + b"\0" * 4 + struct.pack(">I", 0) + b"\0" * 8 + b"\0" * 8
              + struct.pack(">I", 0) + b"".join(_s15(c) for c in D50_XYZ) + b"eNeg")
    header += b"\0" * (128 - len(header))
    return header + struct.pack(">I", len(tags)) + entries + data


# --------------------------------------------------------------------------- embedding

def embed_jpeg(jpeg: bytes, icc: bytes) -> bytes:
    """Insert the profile as APP2 ICC_PROFILE segment(s) right after SOI / APP0."""
    assert jpeg[:2] == b"\xff\xd8"
    pos = 2
    if jpeg[2:4] == b"\xff\xe0":                  # keep JFIF APP0 first
        pos = 4 + struct.unpack(">H", jpeg[4:6])[0]
    chunks = [icc[i:i + 65519] for i in range(0, len(icc), 65519)]
    seg = b"".join(b"\xff\xe2" + struct.pack(">H", 2 + 14 + len(c)) + b"ICC_PROFILE\0"
                   + bytes([n + 1, len(chunks)]) + c for n, c in enumerate(chunks))
    return jpeg[:pos] + seg + jpeg[pos:]


def embed_png(png: bytes, icc: bytes, name: str = "ICC profile") -> bytes:
    """Insert an iCCP chunk right after IHDR."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    ihdr_end = 8 + 8 + struct.unpack(">I", png[8:12])[0] + 4
    body = name.encode("latin-1")[:79] + b"\0\0" + zlib.compress(icc)
    chunk = struct.pack(">I", len(body)) + b"iCCP" + body + struct.pack(">I", zlib.crc32(b"iCCP" + body))
    return png[:ihdr_end] + chunk + png[ihdr_end:]
