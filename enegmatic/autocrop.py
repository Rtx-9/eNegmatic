"""Automatic frame detection for scanned / camera-digitised negatives.

The picture area of a frame is bounded by straight edges that run (almost) the
whole length of the frame: picture -> clear film between frames / rebate,
picture -> film holder, picture -> carrier glass edge, and so on. Picture
content rarely has edges that are both frame-long and exactly parallel to the
frame, and when it does (a horizon, a rink board) the area beyond it is
textured, whereas the area beyond a real frame edge is a uniform band.

So, working in density (log) space:
  1. Find the tilt that makes frame-length edges sharpest (signed gradient
     projections only add up coherently for straight, aligned edges).
  2. On the straightened image, list candidate edges on each side.
  3. Walking outwards from the centre, take the first candidate that spans
     the frame and has a uniform band behind it.
This works the same for colour and B&W film and does not care whether the
picture itself is mostly thin (night scenes) or dense.
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import LUMA

ANGLE_SIZE = 600     # long edge used for the tilt search
EDGE_SIZE = 1200     # long edge used to locate edges


def _density(neg: np.ndarray, size: int) -> np.ndarray:
    h, w = neg.shape[:2]
    s = min(1.0, size / max(h, w))
    img = cv2.resize(neg, (max(1, round(w * s)), max(1, round(h * s))),
                     interpolation=cv2.INTER_AREA) if s < 1 else neg
    lum = img.astype(np.float32) @ LUMA if img.ndim == 3 else img.astype(np.float32)
    d = -np.log10(np.maximum(lum, 1e-4))
    return cv2.GaussianBlur(d, (0, 0), 1.0)


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _gradients(d: np.ndarray):
    return (cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))


def _edge_energy(d: np.ndarray) -> float:
    gx, gy = _gradients(d)
    h, w = d.shape
    px = gx[int(h * .15):int(h * .85)].mean(axis=0)
    py = gy[:, int(w * .15):int(w * .85)].mean(axis=1)
    return float((px ** 2).sum() + (py ** 2).sum())


def _find_angle(d: np.ndarray, max_angle: float) -> float:
    best_a, best_e = 0.0, _edge_energy(d)
    for a in np.arange(-max_angle, max_angle + 1e-6, 0.25):
        e = _edge_energy(_rotate(d, a))
        if e > best_e:
            best_a, best_e = float(a), e
    coarse = best_a
    for a in np.arange(coarse - 0.25, coarse + 0.2501, 0.05):
        e = _edge_energy(_rotate(d, a))
        if e > best_e:
            best_a, best_e = float(a), e
    return best_a


MIN_EDGE = 0.2       # weakest projected edge (density gradient) taken as a frame edge


def _candidates(p: np.ndarray, rel: float = 0.08) -> list[int]:
    mag = np.abs(p)
    thr = max(rel * float(mag.max()), MIN_EDGE)
    n = len(mag)
    m = max(3, int(n * 0.01))   # ignore the scan's own border
    return [i for i in range(m, n - m)
            if mag[i] >= thr and mag[i] == mag[max(0, i - 4):i + 5].max()]


def _level(D: np.ndarray, span, i: int, side: int, width: int = 6) -> float:
    """Median density of a strip just beside position i (side -1 = lower index)."""
    a, b = span
    n = D.shape[1]
    lo, hi = (i - 3 - width, i - 3) if side < 0 else (i + 4, i + 4 + width)
    lo, hi = max(0, lo), min(n, hi)
    return float(np.median(D[a:b, lo:hi])) if hi > lo else float("nan")


def _is_line(D, p, cands, i, span) -> bool:
    """True if i is one side of a thin line in the picture (density steps and returns)."""
    n = D.shape[1]
    for j in cands:
        if j != i and abs(j - i) <= 0.03 * n and np.sign(p[j]) != np.sign(p[i]):
            lo, hi = min(i, j), max(i, j)
            if abs(_level(D, span, lo, -1) - _level(D, span, hi, +1)) < 0.15:
                return True
    return False


DEBUG: list | None = None


def _choose(D: np.ndarray, G: np.ndarray, p: np.ndarray, cands: list[int], all_cands: list[int],
            outward: int, span: tuple[int, int], ref: float) -> int | None:
    """First candidate (inner -> outer) that spans the frame with a uniform band behind it."""
    a, b = span
    n = D.shape[1]

    def coverage(i):
        line = G[a:b, max(0, i - 1):i + 2] * np.sign(p[i])
        return float(np.mean(line.max(axis=1) > 0.3 * abs(p[i])))

    for k, i in enumerate(cands):
        nxt = cands[k + 1] if k + 1 < len(cands) else (-1 if outward < 0 else n)
        band = max(3, min(14, abs(nxt - i) - 4))
        lo, hi = (i - 2 - band, i - 2) if outward < 0 else (i + 3, i + 3 + band)
        strip = D[a:b, max(0, lo):min(n, hi)]
        # Spread along the edge in linear light, relative to the brighter of the band
        # and the whole picture: a dark holder band counts as uniform even though it
        # is noisy in log space. IQR ignores edge printing, dust and frame numbers.
        lum = np.power(10.0, -strip)
        q75, q25 = np.percentile(lum, [75, 25], axis=0)
        spread = float(np.median((q75 - q25) / np.maximum(np.median(lum, axis=0), ref)))
        cov = coverage(i)
        # Where the picture is about as thin as the film base, a picture -> rebate edge
        # fades out, so a partial edge is accepted when a narrow uniform band (rebate,
        # gap, holder lip) separates it from a strong full-length edge just beyond.
        banded = (0 <= nxt < n and abs(nxt - i) <= 0.08 * n and coverage(nxt) >= 0.85)
        if DEBUG is not None:
            DEBUG.append(f"  {'<' if outward < 0 else '>'} {i}: p={p[i]:+.3f} "
                         f"spread={spread:.3f} cov={cov:.2f} banded={banded}")
        if spread > 0.12 or cov < (0.4 if banded else 0.7):
            continue
        if _is_line(D, p, all_cands, i, span):
            continue
        return i
    return None


def _edges(D: np.ndarray, G: np.ndarray, span: tuple[int, int], ref: float):
    """Left / right picture edges (columns) of D, using rows within span."""
    a, b = span
    p = G[a:b].mean(axis=0)
    c = D.shape[1] // 2
    cands = _candidates(p)
    lo = _choose(D, G, p, [i for i in cands if i < c][::-1], cands, -1, span, ref)
    hi = _choose(D, G, p, [i for i in cands if i > c], cands, +1, span, ref)
    return lo, hi


def find_rect(d: np.ndarray):
    """Picture rectangle (x0, y0, x1, y1) in pixels of a straightened density image."""
    h, w = d.shape
    gx, gy = _gradients(d)
    dT, gyT = np.ascontiguousarray(d.T), np.ascontiguousarray(gy.T)
    ys, xs = (int(h * .2), int(h * .8)), (int(w * .2), int(w * .8))
    ref = float(np.median(np.power(10.0, -d[ys[0]:ys[1], xs[0]:xs[1]])))
    rect = None
    for _ in range(2):
        x0, x1 = _edges(d, gx, ys, ref)
        y0, y1 = _edges(dT, gyT, xs, ref)
        rect = (x0, y0, x1, y1)
        fx0, fy0 = x0 if x0 is not None else 0, y0 if y0 is not None else 0
        fx1, fy1 = x1 if x1 is not None else w, y1 if y1 is not None else h
        # second pass: measure along the found frame only
        mx, my = int((fx1 - fx0) * 0.06), int((fy1 - fy0) * 0.06)
        if fx1 - fx0 < w * 0.3 or fy1 - fy0 < h * 0.3:
            break
        xs, ys = (fx0 + mx, fx1 - mx), (fy0 + my, fy1 - my)
    return rect


def detect_frame(neg: np.ndarray, inset: float = 0.01, max_angle: float = 4.0):
    """Return (straighten angle in degrees, normalised crop or None) for an oriented negative."""
    angle = _find_angle(_density(neg, ANGLE_SIZE), max_angle)
    if abs(angle) < 0.05:
        angle = 0.0
    d = _density(neg, EDGE_SIZE)
    if angle:
        d = _rotate(d, angle)
    x0, y0, x1, y1 = find_rect(d)
    if all(v is None for v in (x0, y0, x1, y1)):
        return 0.0, None
    h, w = d.shape
    x0 = (x0 + 1) / w if x0 is not None else 0.0
    y0 = (y0 + 1) / h if y0 is not None else 0.0
    x1 = x1 / w if x1 is not None else 1.0
    y1 = y1 / h if y1 is not None else 1.0
    cw, ch = x1 - x0, y1 - y0
    if cw * ch < 0.25:
        return 0.0, None
    crop = [x0 + inset * cw, y0 + inset * ch, x1 - inset * cw, y1 - inset * ch]
    return round(angle, 2), [round(min(max(c, 0.0), 1.0), 5) for c in crop]
