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

from dataclasses import dataclass

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


def _candidates(p: np.ndarray, rel: float = 0.08, min_edge: float = MIN_EDGE) -> list[int]:
    mag = np.abs(p)
    thr = max(rel * float(mag.max()), min_edge)
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


@dataclass
class _Levels:
    """Reference densities of the scan, used to recognise frame-border material."""
    ref: float        # median linear light of the centre (for relative spreads)
    base: float       # thinnest film density (clear base between frames / rebate)
    picture_hi: float  # densest typical picture density (centre of the scan)

    @property
    def range(self) -> float:
        # density range of the picture; thresholds scale with it so linear raws and
        # gamma-encoded scans (whose densities are compressed) behave the same
        return max(self.picture_hi - self.base, 0.15)


def _border_kind(d_out: float, d_in: float, lv: _Levels) -> str | None:
    """What lies beyond an edge, if it looks like frame border rather than picture.

    Border material is one of: clear film (gaps, rebate) at about base density;
    an (almost) opaque holder, denser than the picture itself; or light shining past
    the film. Skies, walls and other picture areas have in-between densities.
    """
    r = lv.range
    if 10.0 ** -d_out >= 0.9:
        return "light"
    if d_out <= lv.base + 0.2 * r and d_out < d_in:
        return "clear"
    if d_out >= lv.picture_hi + 0.15 * r and d_out > d_in + 0.15 * r:
        return "holder"
    return None


def _choose(D: np.ndarray, G: np.ndarray, p: np.ndarray, cands: list[int], all_cands: list[int],
            weak: list[int], outward: int, span: tuple[int, int], lv: _Levels) -> int | None:
    """First candidate (inner -> outer) that spans the frame with a uniform band of frame
    border (clear film, holder or light) behind it."""
    a, b = span
    n = D.shape[1]

    def coverage(i):
        line = G[a:b, max(0, i - 1):i + 2] * np.sign(p[i])
        return float(np.mean(line.max(axis=1) > 0.3 * abs(p[i])))

    def strip_between(i, nxt):
        band = max(3, min(14, abs(nxt - i) - 4))
        lo, hi = (i - 2 - band, i - 2) if outward < 0 else (i + 3, i + 3 + band)
        strip = D[a:b, max(0, lo):min(n, hi)]
        return strip, (float(np.median(strip)) if strip.size else float("nan"))

    def outside(k):
        """Strip just beyond candidate k (up to the next candidate), and its median density."""
        nxt = cands[k + 1] if k + 1 < len(cands) else (-1 if outward < 0 else n)
        return strip_between(cands[k], nxt)

    def beyond(i):
        """Frame-long edges further out (including faint ones) with the density behind each."""
        pos = sorted((j for j in weak if (j - i) * outward > 3 and abs(j - i) <= 0.25 * n),
                     key=lambda j: abs(j - i))
        for idx, j in enumerate(pos):
            nxt = pos[idx + 1] if idx + 1 < len(pos) else (-1 if outward < 0 else n)
            if coverage(j) >= 0.85:
                yield j, strip_between(j, nxt)[1]

    def inside(i):
        inner = D[a:b, max(0, i - 17):max(1, i - 3)] if outward > 0 else D[a:b, i + 4:i + 18]
        return float(np.median(inner)) if inner.size else float("nan")

    for k, i in enumerate(cands):
        nxt = cands[k + 1] if k + 1 < len(cands) else (-1 if outward < 0 else n)
        strip, d_out = outside(k)
        # Spread along the edge in linear light, relative to the brighter of the band
        # and the whole picture: a dark holder band counts as uniform even though it
        # is noisy in log space. IQR ignores edge printing, dust and frame numbers.
        lum = np.power(10.0, -strip)
        q75, q25 = np.percentile(lum, [75, 25], axis=0)
        spread = float(np.median((q75 - q25) / np.maximum(np.median(lum, axis=0), lv.ref)))
        cov = coverage(i)
        # Where the picture is about as thin as the film base, a picture -> rebate edge
        # fades out, so a partial edge is accepted when a narrow uniform band (rebate,
        # gap, holder lip) separates it from a strong full-length edge just beyond.
        banded = (0 <= nxt < n and abs(nxt - i) <= 0.08 * n and coverage(nxt) >= 0.85)
        d_in = inside(i)
        kind = _border_kind(d_out, d_in, lv)
        if kind is not None:
            # A flat area of the picture (clear sky, deep shadow) right at the frame edge
            # can look like border material. But real border is the last thing before the
            # film edge: if another frame-long edge just beyond still has clear film or
            # light behind it, this band is part of the picture. (Sprocket holes are not
            # frame-long, so they don't count.)
            prev = d_out
            for j, d_j in beyond(i):
                if _border_kind(d_j, prev, lv) in ("clear", "light") \
                        and (kind == "holder" or d_j <= d_out + 0.05 * lv.range):
                    kind = None
                    break
                # A wide dense band (bright sky) with a clearly denser holder beyond it is
                # picture too. Narrow ones are holder lips / out-of-focus holder edges.
                if kind == "holder" and abs(nxt - i) > 0.03 * n and d_j >= d_out + 0.15 * lv.range:
                    kind = None
                    break
                prev = d_j
        if kind is None and 0 <= nxt < n and abs(nxt - i) <= 0.03 * n and coverage(nxt) >= 0.85:
            # narrow band (holder lip, rebate strip) right before a real border edge
            if _border_kind(outside(k + 1)[1], d_out, lv) is not None:
                kind = "band"
        if DEBUG is not None:
            DEBUG.append(f"  {'<' if outward < 0 else '>'} {i}: p={p[i]:+.3f} spread={spread:.3f} "
                         f"cov={cov:.2f} banded={banded} out={d_out:.2f} in={d_in:.2f} kind={kind}")
        if spread > 0.12 or cov < (0.4 if banded else 0.7) or kind is None:
            continue
        if _is_line(D, p, all_cands, i, span):
            continue
        return i
    return None


def _edges(D: np.ndarray, G: np.ndarray, span: tuple[int, int], lv: _Levels):
    """Left / right picture edges (columns) of D, using rows within span."""
    a, b = span
    p = G[a:b].mean(axis=0)
    c = D.shape[1] // 2
    cands = _candidates(p)
    weak = _candidates(p, rel=0.02, min_edge=0.04)     # faint edges, for the look-ahead only
    lo = _choose(D, G, p, [i for i in cands if i < c][::-1], cands, weak, -1, span, lv)
    hi = _choose(D, G, p, [i for i in cands if i > c], cands, weak, +1, span, lv)
    return lo, hi


def _levels(d: np.ndarray) -> _Levels:
    h, w = d.shape
    centre = d[int(h * .2):int(h * .8), int(w * .2):int(w * .8)]
    film = d[d > 0.05]                       # ignore light shining past the film
    base = float(np.percentile(film, 0.5)) if film.size > 100 else float(d.min())
    return _Levels(ref=float(np.median(np.power(10.0, -centre))), base=base,
                   picture_hi=float(np.percentile(centre, 99)))


def find_rect(d: np.ndarray):
    """Picture rectangle (x0, y0, x1, y1) in pixels of a straightened density image."""
    h, w = d.shape
    gx, gy = _gradients(d)
    dT, gyT = np.ascontiguousarray(d.T), np.ascontiguousarray(gy.T)
    ys, xs = (int(h * .2), int(h * .8)), (int(w * .2), int(w * .8))
    lv = _levels(d)
    rect = None
    for _ in range(2):
        x0, x1 = _edges(d, gx, ys, lv)
        y0, y1 = _edges(dT, gyT, xs, lv)
        rect = (x0, y0, x1, y1)
        fx0, fy0 = x0 if x0 is not None else 0, y0 if y0 is not None else 0
        fx1, fy1 = x1 if x1 is not None else w, y1 if y1 is not None else h
        # second pass: measure along the found frame only
        mx, my = int((fx1 - fx0) * 0.06), int((fy1 - fy0) * 0.06)
        if fx1 - fx0 < w * 0.3 or fy1 - fy0 < h * 0.3:
            break
        xs, ys = (fx0 + mx, fx1 - mx), (fy0 + my, fy1 - my)
    return rect


REFINE_SIZE = 2000   # long edge used to trace frame edges for the precise angle


def _edge_points(g: np.ndarray, pos: float, lo: int, hi: int, step: int = 2):
    """Sub-pixel location of a straight edge near column `pos` for rows lo..hi of gradient g.

    Returns (rows, columns, strengths). Columns come from a parabola through the gradient
    peak and its neighbours, so the edge is located to a fraction of a pixel.
    """
    w = g.shape[1]
    c = int(round(pos))
    if c - 8 < 1 or c + 8 >= w - 1:
        return np.empty(0), np.empty(0), np.empty(0)
    win = g[lo:hi:step, c - 8:c + 9]
    sign = np.sign(win[:, 6:11].mean()) or 1.0
    win = win * sign
    k = np.clip(win.argmax(axis=1), 1, win.shape[1] - 2)
    r = np.arange(win.shape[0])
    y0, y1, y2 = win[r, k - 1], win[r, k], win[r, k + 1]
    denom = y0 - 2 * y1 + y2
    off = np.where(np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / np.where(denom == 0, 1, denom), 0.0)
    cols = c - 8 + k + np.clip(off, -1, 1)
    rows = np.arange(lo, hi, step)[: len(cols)]
    return rows.astype(np.float64), cols.astype(np.float64), y1


def _fit_edge(rows, cols, strength):
    """Robust line fit col = m*row + b; returns (slope, inlier count, rms residual) or None."""
    keep = strength > 0.35 * np.median(strength) if len(strength) else strength
    if keep.sum() < 20:
        return None
    rows, cols = rows[keep], cols[keep]
    sel = np.ones(len(rows), bool)
    for _ in range(4):
        if sel.sum() < 20:
            return None
        m, b = np.polyfit(rows[sel], cols[sel], 1)
        res = np.abs(cols - (m * rows + b))
        sel = res < max(0.6, 2.5 * np.median(res[sel]))
    if sel.mean() < 0.5:
        return None
    m, b = np.polyfit(rows[sel], cols[sel], 1)
    rms = float(np.sqrt(np.mean((cols[sel] - (m * rows[sel] + b)) ** 2)))
    return float(m), int(sel.sum()), rms


def _refine_angle(neg: np.ndarray, angle: float, rect, shape) -> float | None:
    """Measure the remaining tilt from the detected frame edges themselves.

    The edges are traced at REFINE_SIZE with sub-pixel precision and fitted with straight
    lines; their slopes give the residual tilt (typically to ~0.01 deg).
    """
    d = _density(neg, REFINE_SIZE)
    if angle:
        d = _rotate(d, angle)
    h, w = d.shape
    sy, sx = h / shape[0], w / shape[1]
    gx, gy = _gradients(d)
    x0, y0, x1, y1 = rect
    fy0, fy1 = (y0 if y0 is not None else 0) * sy, (y1 if y1 is not None else shape[0]) * sy
    fx0, fx1 = (x0 if x0 is not None else 0) * sx, (x1 if x1 is not None else shape[1]) * sx
    # vertical edges: x = m*y + b  ->  tilt = atan(m)
    vertical = []
    for x in (x0, x1):
        if x is None:
            continue
        lo, hi = int(fy0 + 0.1 * (fy1 - fy0)), int(fy1 - 0.1 * (fy1 - fy0))
        fit = _fit_edge(*_edge_points(gx, x * sx, lo, hi))
        if fit and fit[2] < 1.0:               # only cleanly straight edges
            vertical.append(np.degrees(np.arctan(fit[0])))
    # horizontal edges: y = m*x + b  ->  tilt = -atan(m)
    horizontal = []
    gyT = np.ascontiguousarray(gy.T)
    for y in (y0, y1):
        if y is None:
            continue
        lo, hi = int(fx0 + 0.1 * (fx1 - fx0)), int(fx1 - 0.1 * (fx1 - fx0))
        fit = _fit_edge(*_edge_points(gyT, y * sy, lo, hi))
        if fit and fit[2] < 1.0:
            horizontal.append(-np.degrees(np.arctan(fit[0])))
    # Scanned frames are rarely perfect rectangles: keystoning (camera not exactly parallel
    # to the film), lens distortion and film curl tilt opposite edges in opposite ways by a
    # few tenths of a degree. Averaging each pair of opposite edges cancels that, and
    # averaging the pairs gives the rotation that leaves the frame as level as possible.
    pairs = [float(np.mean(t)) for t in (vertical, horizontal) if t]
    if not pairs:
        return None
    return angle - float(np.mean(pairs))


def detect_frame(neg: np.ndarray, inset: float = 0.01, max_angle: float = 4.0):
    """Return (straighten angle in degrees, normalised crop or None) for an oriented negative."""
    angle = _find_angle(_density(neg, ANGLE_SIZE), max_angle)
    d0 = _density(neg, EDGE_SIZE)
    d = _rotate(d0, angle) if angle else d0
    x0, y0, x1, y1 = find_rect(d)
    if any(v is not None for v in (x0, y0, x1, y1)):
        refined = _refine_angle(neg, angle, (x0, y0, x1, y1), d.shape)
        if refined is not None and abs(refined - angle) < 0.5:
            angle = refined
            d = _rotate(d0, angle) if abs(angle) >= 0.005 else d0
            x0, y0, x1, y1 = find_rect(d)
    if abs(angle) < 0.01:
        angle = 0.0
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
    return round(angle, 3), [round(min(max(c, 0.0), 1.0), 5) for c in crop]



STRIP_MAX = 0.06        # widest unexposed strip removed, fraction of the crop
STRIP_LEVEL = 0.07      # strip density above the film base (unexposed film + fog)
STRIP_FLAT = 0.05       # spread (p90 - p10) across a strip line: plain film, no picture
STRIP_RISE = 0.03       # sustained density rise that marks the start of the picture


def trim_unexposed(neg: np.ndarray, angle: float, crop) -> list:
    """Remove strips of unexposed film between the picture and the detected border.

    Many cameras leave a narrow strip of clear film at the ends of the frame, and the
    frame search stops at the holder beyond it. The strip is film base all the way
    across, so a line of it is uniformly at base density; the picture starts where the
    density rises above that for good. Where the picture next to the strip is itself
    as thin as the base (black night sky), nothing is removed.
    """
    d = _density(neg, EDGE_SIZE)
    if angle:
        d = _rotate(d, angle)
    h, w = d.shape
    lv = _levels(d)
    box = [crop[0] * w, crop[1] * h, crop[2] * w, crop[3] * h]

    def strip(D, lo, hi, a, b, inward):
        """Width of the unexposed strip at line `lo` (inward +1) or `hi` (-1)."""
        ra, rb = int(a + 0.1 * (b - a)), int(b - 0.1 * (b - a))
        if rb - ra < 10:
            return 0
        start = int(round(lo)) if inward > 0 else int(round(hi)) - 1
        limit = int(STRIP_MAX * (hi - lo))
        idx = [start + inward * k for k in range(limit + 8)]
        idx = [i for i in idx if 0 <= i < D.shape[1]]
        if len(idx) < 12:
            return 0
        lines = D[ra:rb, idx]
        p10, p90 = np.percentile(lines, [10, 90], axis=0)
        above = p90 - lv.base
        flat = (above <= STRIP_LEVEL) & (p90 - p10 <= STRIP_FLAT)
        k = 0
        while k < 6 and k < len(idx) and not flat[k]:   # the edge itself is blurred
            k += 1
        first = k
        while k < len(idx) and flat[k]:
            k += 1
        if k - first < 4 or k > limit:                  # no strip, or strip runs on (dark picture)
            return 0
        level = float(np.median(above[first:k]))
        after = above[k + 2:k + 8]                      # past the soft picture edge
        if len(after) < 4 or float(np.min(after)) < level + STRIP_RISE:
            return 0
        return k + 1

    x0, y0, x1, y1 = box
    l_ = strip(d, x0, x1, y0, y1, +1)
    r_ = strip(d, x0, x1, y0, y1, -1)
    dT = np.ascontiguousarray(d.T)
    t_ = strip(dT, y0, y1, x0, x1, +1)
    b_ = strip(dT, y0, y1, x0, x1, -1)
    x0, x1, y0, y1 = x0 + l_, x1 - r_, y0 + t_, y1 - b_
    return [round(float(v), 6) for v in (x0 / w, y0 / h, x1 / w, y1 / h)]
