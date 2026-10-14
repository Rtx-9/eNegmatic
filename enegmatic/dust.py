"""Dust and hair removal on the negative, before any geometry or tone processing.

Dust sitting on the film blocks light, so on the negative it is a small, sharp patch of
extra density on top of the picture. That makes it a white speck (or white curly line for
hairs) in the positive. Automatic detection looks for exactly that:

  * a local density excess over the background (white top-hat), well above the grain;
  * thin: specks and fibres, nothing wider than a few tenths of a percent of the frame;
  * on a smooth background (sky, walls, skin), where dust is visible - textured areas are
    left alone, so fine picture detail is never mistaken for dust;
  * neutral on colour film: dust darkens all three dye layers equally, while small coloured
    lights and highlights do not.

Manual spots (the healing brush) are stored per frame as [x, y, r] in normalised
coordinates of the original, unrotated negative; r is a fraction of its long edge.

Removal fills the masked pixels from the surroundings in density space (OpenCV Telea on a
smoothed copy) and puts grain back by borrowing the fine texture of a nearby clean patch,
so healed spots don't look smeared.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

DETECT_SIZE = 4000      # long edge the automatic detection runs at (or less)
EPS = 1e-5


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def _density(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint16:
        x = x.astype(np.float32) * np.float32(1.0 / 65535.0)
    return -np.log10(np.maximum(x.astype(np.float32, copy=False), EPS))


# --------------------------------------------------------------------------- detection

def detect(neg: np.ndarray, strength: float = 50.0, color: bool = True) -> np.ndarray:
    """Boolean mask (at the negative's resolution) of dust specks and hairs.

    strength 0..100: higher removes fainter / less certain dust.
    """
    h, w = neg.shape[:2]
    scale = min(1.0, DETECT_SIZE / max(h, w))
    small = neg
    if scale < 1.0:
        small = cv2.resize(neg, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    m = _detect(small, float(strength), color)
    if scale < 1.0:
        m = cv2.resize(m.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR) > 40
    return m


BUSY_MAX = (0.025, 0.055)   # busy-surroundings limit at strength 0 / 100
EDGE_MAX = (0.8, 2.2)       # background-structure limit (grain sigmas) at strength 0 / 100
CROWD_RADIUS = 0.03         # B&W: neighbourhood (fraction of the long edge) ...
CROWD_MAX = 3               # ... in which this many other specks means lights, not dust


def _noise_model(r: np.ndarray, level: np.ndarray, valid: np.ndarray, step: int):
    """Offset and grain sigma of residual r as functions of the background level.

    Grain noise in density grows with density (less light gets through), so a single
    global sigma would flag grain in dense areas and miss dust in thin ones.
    """
    rs, lv = r[::step, ::step], level[::step, ::step]
    vs = valid[::step, ::step]
    if vs.sum() > 1000:
        rs, lv = rs[vs], lv[vs]
    rs, lv = rs.ravel(), lv.ravel()
    edges = np.unique(np.percentile(lv, np.linspace(0, 100, 13)))
    centers, meds, sigs = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (lv >= a) & (lv <= b)
        if sel.sum() < 200:
            continue
        x = rs[sel]
        m = float(np.median(x))
        centers.append(float(np.median(lv[sel])))
        meds.append(m)
        sigs.append(max(1.4826 * float(np.median(np.abs(x - m))), 1e-4))
    if not centers:
        m = float(np.median(rs))
        centers, meds = [0.0], [m]
        sigs = [max(1.4826 * float(np.median(np.abs(rs - m))), 1e-4)]
    c = np.asarray(centers, np.float32)
    mm, ss = np.asarray(meds, np.float32), np.asarray(sigs, np.float32)
    return (lambda x: np.interp(x, c, mm).astype(np.float32),
            lambda x: np.interp(x, c, ss).astype(np.float32))


def _features(neg: np.ndarray, strength: float, color: bool) -> dict:
    """Candidate specks and the measurements the detector decides on."""
    h, w = neg.shape[:2]
    L = float(max(h, w))
    d = _density(neg)
    if d.ndim == 2:
        d = d[..., None]
    if not color or d.shape[2] == 1:
        d = d.mean(axis=2, keepdims=True)
    nch = d.shape[2]
    lum = d.mean(axis=2)
    step = max(1, int(math.sqrt(h * w / 400_000)))

    # Only look inside the picture: the opaque holder / unexposed surround sits far
    # above any image density and is pure sensor noise.
    base = float(np.percentile(lum[::step, ::step], 0.5))
    k = _odd(L * 0.006)                             # widest dust the top-hat isolates
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    valid = cv2.erode((lum < base + 2.8).astype(np.uint8), kern) > 0

    # Per-channel white top-hat, each in units of its own grain noise at that density.
    # The blue layer of a colour negative is far noisier than red and green, so equal
    # weighting of the channels would bury the dust.
    s0 = max(0.6, L / 4000.0)                       # grain pre-smoothing
    z = np.zeros((h, w), np.float32)
    res = []
    for c in range(nch):
        dc = cv2.GaussianBlur(d[..., c], (0, 0), s0)
        bc = cv2.blur(cv2.morphologyEx(dc, cv2.MORPH_OPEN, kern), (k, k))
        rc = dc - bc
        off, sig = _noise_model(rc, bc, valid, step)
        rc -= off(bc)
        z += rc / sig(bc)
        res.append(rc)
    z /= math.sqrt(nch)
    z[~valid] = 0.0

    st = min(max(strength, 0.0), 100.0) / 100.0
    k_hi = 11.0 - 6.0 * st                          # seed threshold, grain sigmas
    k_lo = max(2.5, 0.4 * k_hi)
    region = (z > k_lo).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)

    flat = lab.ravel()
    sel = flat > 0
    idx = flat[sel]
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, idx, z.ravel()[sel])
    dist = cv2.distanceTransform(region, cv2.DIST_L2, 3)
    width = np.zeros(n, np.float32)
    np.maximum.at(width, idx, dist.ravel()[sel])
    width *= 2.0

    # How busy the surroundings are: the fraction of pixels in a window a few dust
    # widths wide that deviate strongly from the background either way, not counting
    # candidate specks. Grain alone gives a few percent; foliage, crowds, text and
    # other fine detail give far more. Dust is only removed where it stands alone.
    win = (_odd(k * 3),) * 2
    dev = ((np.abs(z) > 2.5) & (region == 0)).astype(np.float32)
    busy_map = cv2.blur(dev, win)
    area1 = np.maximum(area, 1)
    busy = np.bincount(idx, weights=busy_map.ravel()[sel], minlength=n) / area1
    # Broader structure the top-hat keeps (edges, gradients), in grain sigmas.
    lb = cv2.blur(cv2.morphologyEx(cv2.GaussianBlur(lum, (0, 0), s0), cv2.MORPH_OPEN, kern), (k, k))
    lres = cv2.GaussianBlur(lum, (0, 0), s0) - lb
    lsig = _noise_model(lres, lb, valid, step)[1](lb)
    mu = cv2.blur(lb, win)
    edge_map = np.sqrt(np.maximum(cv2.blur(lb * lb, win) - mu * mu, 0.0)) / lsig
    edge = np.bincount(idx, weights=edge_map.ravel()[sel], minlength=n) / area1

    exc = np.stack([np.bincount(idx, weights=rc.ravel()[sel], minlength=n) / area1
                    for rc in res], axis=1)
    yy, xx = np.divmod(np.flatnonzero(sel), w)
    centroid = np.stack([np.bincount(idx, weights=xx, minlength=n) / area1 / w,
                         np.bincount(idx, weights=yy, minlength=n) / area1 / h], axis=1)

    # Shape: hairs curl, while long straight bright lines are nearly always part of the
    # picture (stripes, wires, edges of buildings). straight = minor / major axis.
    sx, sy = centroid[:, 0] * w, centroid[:, 1] * h
    vxx = np.bincount(idx, weights=xx * xx, minlength=n) / area1 - sx * sx
    vyy = np.bincount(idx, weights=yy * yy, minlength=n) / area1 - sy * sy
    vxy = np.bincount(idx, weights=xx * yy, minlength=n) / area1 - sx * sy
    half_tr = (vxx + vyy) / 2.0
    disc = np.sqrt(np.maximum(half_tr * half_tr - (vxx * vyy - vxy * vxy), 0.0))
    straight = np.sqrt(np.maximum(half_tr - disc, 0.0) / np.maximum(half_tr + disc, 1e-6))
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]).astype(np.float32)
    return dict(lab=lab, n=n, peak=peak, width=width, area=area, busy=busy, edge=edge, exc=exc,
                straight=straight, extent=extent,
                centroid=centroid, k_hi=k_hi, st=st, L=L, s0=s0)


def _detect(neg: np.ndarray, strength: float, color: bool) -> np.ndarray:
    f = _features(neg, strength, color)
    n, st, k_hi, L = f["n"], f["st"], f["k_hi"], f["L"]
    if n <= 1:
        return np.zeros(neg.shape[:2], bool)
    keep = f["peak"] > k_hi
    keep &= f["width"] <= max(3.0, L * 0.004)
    keep &= f["area"] <= (L * 0.012) ** 2
    keep &= f["busy"] <= BUSY_MAX[0] + (BUSY_MAX[1] - BUSY_MAX[0]) * st
    keep &= f["edge"] <= EDGE_MAX[0] + (EDGE_MAX[1] - EDGE_MAX[0]) * st
    keep &= f["extent"] <= L * 0.08                              # longest hair
    keep &= ~((f["extent"] > L * 0.015) & (f["straight"] < 0.15))   # straight lines: picture
    keep[0] = False

    exc = f["exc"]
    if exc.shape[1] == 3 and keep.any():
        # Dust blocks light in all three layers; its density excess has one colour
        # signature per scan (set by the film and light source). Learn it from the
        # clearest specks and drop candidates that don't match, such as small
        # coloured lights and highlights.
        pos = np.all(exc > 0, axis=1)
        unit = exc / np.maximum(np.linalg.norm(exc, axis=1, keepdims=True), 1e-9)
        strong = keep & pos & (f["peak"] > 2.0 * k_hi)
        keep &= pos
        if strong.sum() >= 6:
            ref = np.median(unit[strong], axis=0)
            ref /= np.linalg.norm(ref)
            keep &= unit @ ref >= 0.97 - 0.07 * st
    elif keep.sum() > 3:
        # B&W has no colour signature to tell dust from small lights. Lamps, though,
        # come in rows and clusters (ceilings, windows, stadium lights), while dust is
        # scattered: drop specks with several others crowded close around them.
        ids = np.flatnonzero(keep)
        pts = f["centroid"][ids] * np.array(f["lab"].shape[::-1], np.float32)
        dd = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        crowd = (dd < CROWD_RADIUS * L).sum(axis=1) - 1
        keep[ids[crowd >= CROWD_MAX]] = False

    mask = keep[f["lab"]]
    grow = _odd(2 * f["s0"] + 2)                     # cover the soft rim of each speck
    mask = cv2.dilate(mask.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))) > 0
    return mask


# --------------------------------------------------------------------------- manual spots

def spots_mask(shape, spots) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    L = max(h, w)
    for sp in spots or ():
        x, y, r = (float(v) for v in sp[:3])
        cv2.circle(m, (int(round(x * w)), int(round(y * h))), max(1, int(round(r * L))), 1, -1,
                   lineType=cv2.LINE_8)
    return m > 0


# --------------------------------------------------------------------------- healing

def heal(neg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of neg with the masked pixels filled from their surroundings."""
    out = neg.copy()
    if not mask.any():
        return out
    h, w = mask.shape
    L = max(h, w)
    sg = max(1.0, L / 3000.0)                       # grain scale for texture transfer
    m8 = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        pad = max(6, int(0.6 * max(bw, bh)) + int(3 * sg))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        _heal_box(out, mask, x0, y0, x1, y1, sg)
    return out


def _fill(d: np.ndarray, hole: np.ndarray, sg: float) -> np.ndarray:
    """Smooth fill of the hole from the surrounding pixels (normalised convolution).

    Each hole pixel takes the Gaussian-weighted mean of the valid pixels around it, at
    the smallest scale that reaches enough of them, so thin hairs fill from both sides
    and larger spots blend smoothly. (OpenCV's Telea inpaint is biased on float data.)
    """
    w = (~hole).astype(np.float32)
    out = d.copy()
    todo = hole.copy()
    sigma = max(1.0, sg)
    while todo.any():
        den = cv2.GaussianBlur(w, (0, 0), sigma)
        ok = todo & (den > 0.08)
        if ok.any() or sigma > 4 * max(d.shape[:2]):
            ok = todo if not ok.any() else ok
            for c in range(d.shape[2]):
                num = cv2.GaussianBlur(d[..., c] * w, (0, 0), sigma)
                out[..., c][ok] = (num / np.maximum(den, 1e-6))[ok]
            todo &= ~ok
        sigma *= 1.6
    return out


def _heal_box(out, mask, x0, y0, x1, y1, sg):
    """Heal one spot in place. Reads the (already partly healed) output, so
    neighbouring spots never undo each other."""
    src = out
    box_m = mask[y0:y1, x0:x1]
    if not box_m.any():
        return
    d = _density(src[y0:y1, x0:x1])
    if d.ndim == 2:
        d = d[..., None]
    m8 = box_m.astype(np.uint8)
    bh, bw = m8.shape
    # texture donor: the same-size patch beside the box that is least covered by the mask
    donor = None
    H, W = mask.shape
    best = 2.0
    for dx, dy in ((bw, 0), (-bw, 0), (0, bh), (0, -bh), (bw, bh), (-bw, -bh), (bw, -bh), (-bw, bh)):
        sx0, sy0 = x0 + dx, y0 + dy
        if sx0 < 0 or sy0 < 0 or sx0 + bw > W or sy0 + bh > H:
            continue
        cover = float(mask[sy0:sy0 + bh, sx0:sx0 + bw].mean())
        if cover < best:
            best, donor = cover, (sx0, sy0)
            if cover == 0.0:
                break
    filled = _fill(d, m8 > 0, sg)
    if donor is not None and best < 0.5:
        sx0, sy0 = donor
        dd = _density(src[sy0:sy0 + bh, sx0:sx0 + bw])
        if dd.ndim == 2:
            dd = dd[..., None]
        for c in range(d.shape[2]):
            filled[..., c] += dd[..., c] - cv2.GaussianBlur(dd[..., c], (0, 0), sg)
    # feather the seam over a pixel or two
    alpha = cv2.GaussianBlur(m8.astype(np.float32), (0, 0), max(0.7, 0.5 * sg))
    alpha = np.maximum(alpha, m8.astype(np.float32))[..., None]
    res = d * (1.0 - alpha) + filled * alpha
    lin = np.power(10.0, -res, dtype=np.float32)
    if out.dtype == np.uint16:
        lin = np.clip(lin * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    else:
        lin = lin.astype(out.dtype)
    if out.ndim == 2:
        lin = lin[..., 0]
    touched = alpha[..., 0] > 1e-3
    out[y0:y1, x0:x1][touched] = lin[touched]


# --------------------------------------------------------------------------- settings glue

def active(s) -> bool:
    return bool(s.dust_auto or s.dust_spots)


def _auto_key(s) -> tuple:
    keep = tuple(tuple(round(float(v), 5) for v in p[:2]) for p in (s.dust_keep or ()))
    return (round(float(s.dust_strength), 1), s.film_type == "color", keep)


def _spots_key(s) -> tuple:
    return tuple(tuple(round(float(v), 6) for v in sp[:3]) for sp in (s.dust_spots or ()))


def dust_key(s) -> tuple:
    return (_auto_key(s) if s.dust_auto else None, _spots_key(s))


def auto_specks(neg: np.ndarray, s) -> tuple[np.ndarray, list]:
    """Auto-detected mask minus the specks the user chose to keep, and the specks as
    [x, y, r] in normalised source coordinates (r: fraction of the long edge)."""
    mask = detect(neg, s.dust_strength, color=s.film_type == "color")
    h, w = mask.shape
    L = max(h, w)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    drop = set()
    tol = max(3, int(round(L * 0.004)))
    for p in s.dust_keep or ():
        x, y = int(float(p[0]) * w), int(float(p[1]) * h)
        win = lab[max(0, y - tol):y + tol + 1, max(0, x - tol):x + tol + 1]
        drop.update(int(v) for v in np.unique(win) if v)
    if drop:
        mask &= ~np.isin(lab, list(drop))
    specks = [[float(cent[i, 0] / w), float(cent[i, 1] / h),
               float(max(stats[i, 2], stats[i, 3]) / 2.0 / L)]
              for i in range(1, n) if i not in drop]
    return mask, specks


def clean(neg: np.ndarray, s) -> tuple[np.ndarray, list]:
    """Negative with dust removed according to the settings, and the auto specks."""
    if not active(s):
        return neg, []
    mask = np.zeros(neg.shape[:2], bool)
    specks = []
    if s.dust_auto:
        mask, specks = auto_specks(neg, s)
    if s.dust_spots:
        mask |= spots_mask(neg.shape, s.dust_spots)
    return heal(neg, mask), specks


class Cleaner:
    """Caches the cleaned negative in two stages, so that painting with the healing
    brush doesn't redo the automatic detection, and tone edits redo nothing."""

    def __init__(self):
        self._auto_k = self._auto = None        # (key, healed negative, specks)
        self._full_k = self._full = None

    def get(self, neg: np.ndarray, s) -> tuple[np.ndarray, list]:
        if not active(s):
            return neg, []
        ak = _auto_key(s) if s.dust_auto else None
        if self._auto_k != ak or self._auto is None:
            if ak is None:
                self._auto = (neg, [])
            else:
                mask, specks = auto_specks(neg, s)
                self._auto = (heal(neg, mask), specks)
            self._auto_k, self._full_k = ak, None
        fk = (ak, _spots_key(s))
        if self._full_k != fk or self._full is None:
            base, specks = self._auto
            out = heal(base, spots_mask(neg.shape, s.dust_spots)) if s.dust_spots else base
            self._full, self._full_k = (out, specks), fk
        return self._full
