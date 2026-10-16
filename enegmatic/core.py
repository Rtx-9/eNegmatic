"""Negative -> positive conversion pipeline. Pure numpy / OpenCV, no Qt.

The negative is processed in *density* space (-log10 of transmitted light):

    d   = -log10(neg)                       per channel
    t   = (d - dmin) / (dmax - dmin)        per channel -> removes the orange mask
                                            and balances the channels (auto WB)
    t   = t ** wb_gamma                     optional mid-tone grey balance
    ... exposure / white balance / tone / colour adjustments ...

Because density is logarithmic, any pure power-law encoding of the input
(linear raw, gamma-encoded TIFF/JPEG) is cancelled by the per-channel
normalisation, so the same pipeline works for raw files and scanner output.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields

import cv2
import numpy as np

EPS = 1e-5
DISPLAY_GAMMA = 2.2
PRINT_STOPS = 5.5
PRINT_STOPS_BW = 6.5   # B&W prints like a harder paper grade: deeper blacks, more snap
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Fields that change which pixels / statistics the auto analysis looks at.
ANALYSIS_FIELDS = ("film_type", "rotation", "flip_h", "straighten", "crop",
                   "clip_black", "clip_white", "base_rgb", "auto_wb", "auto_brightness")
# Groups used by selective copy / paste and the per-section reset buttons.
SETTING_GROUPS = {
    "film": ("Film type & inversion",
             ("film_type", "clip_black", "clip_white", "base_rgb", "auto_wb", "auto_brightness",
              "lock_analysis", "analysis")),
    "tone": ("Tone", ("exposure", "contrast", "gamma", "black", "white", "highlights", "shadows")),
    "color": ("Colour & white balance", ("wb_gains", "temperature", "tint", "saturation")),
    "detail": ("Detail", ("sharpen", "denoise")),
    "dust": ("Auto dust removal (beta)", ("dust_auto", "dust_strength")),
    "orientation": ("Rotation & flip", ("rotation", "flip_h")),
    "crop": ("Crop & straighten", ("crop", "straighten", "crop_aspect")),
}
DEFAULT_COPY_GROUPS = ("film", "tone", "color", "detail", "orientation")


@dataclass
class Settings:
    film_type: str = "color"            # "color" | "bw"
    # geometry
    rotation: int = 0                   # multiple of 90, counter-clockwise
    flip_h: bool = False
    straighten: float = 0.0             # fine rotation, degrees (CCW positive)
    crop: list | None = None            # normalised [x0, y0, x1, y1] after rotation
    crop_aspect: str = ""               # crop tool aspect, a key of ASPECTS ("" = free)
    # inversion / analysis
    clip_black: float = 0.0             # % of the picture pushed to pure black (beyond auto levels)
    clip_white: float = 0.0             # % of the picture pushed to pure white
    base_rgb: list | None = None        # picked film-base colour (linear negative)
    auto_wb: bool = True                # mid-tone grey balance after inversion
    auto_brightness: bool = True        # lift dark-looking conversions to a normal brightness
    lock_analysis: bool = False         # keep the current analysis (roll-consistent)
    analysis: dict | None = None
    # colour
    wb_gains: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    temperature: float = 0.0
    tint: float = 0.0
    saturation: float = 0.0
    # tone
    exposure: float = 0.0
    contrast: float = 30.0
    gamma: float = 1.0
    black: float = 0.0
    white: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    sharpen: float = 0.0
    denoise: float = 25.0               # 0..100 scan (digital) noise reduction
    # dust removal (see dust.py)
    dust_auto: bool = False             # automatic dust & hair removal
    dust_strength: float = 50.0         # 0..100 sensitivity
    dust_spots: list = field(default_factory=list)   # healing brush: [x, y, r] on the source
    dust_keep: list = field(default_factory=list)    # [x, y]: auto specks the user kept

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        names = {f.name for f in fields(cls)}
        s = cls()
        for k, v in (d or {}).items():
            if k in names:
                setattr(s, k, v)
        return s

    def copy(self) -> "Settings":
        return Settings.from_dict(self.to_dict())


def analysis_key(s: Settings) -> str:
    parts = []
    for name in ANALYSIS_FIELDS:
        v = getattr(s, name)
        if isinstance(v, float):
            v = round(v, 4)
        elif isinstance(v, (list, tuple)):
            v = [round(float(x), 4) for x in v]
        parts.append(v)
    return json.dumps(parts)


# --------------------------------------------------------------------------- geometry

def orient(img: np.ndarray, rotation: int, flip_h: bool) -> np.ndarray:
    k = (int(rotation) // 90) % 4
    if k:
        img = np.rot90(img, k)
    if flip_h:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def straighten(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def crop_pixels(shape, crop) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    if not crop:
        return 0, 0, w, h
    x0, y0, x1, y1 = (min(max(float(c), 0.0), 1.0) for c in crop)
    px0, px1 = int(round(x0 * w)), int(round(x1 * w))
    py0, py1 = int(round(y0 * h)), int(round(y1 * h))
    px1 = max(px1, px0 + 2)
    py1 = max(py1, py0 + 2)
    return px0, py0, min(px1, w), min(py1, h)


def apply_crop(img: np.ndarray, crop) -> np.ndarray:
    x0, y0, x1, y1 = crop_pixels(img.shape, crop)
    return img[y0:y1, x0:x1]


def apply_geometry(neg: np.ndarray, s: Settings, crop: bool = True) -> np.ndarray:
    img = orient(neg, s.rotation, s.flip_h)
    img = straighten(img, s.straighten)
    return apply_crop(img, s.crop) if crop else img


def _crop_rot_ccw(c):
    x0, y0, x1, y1 = c
    return [y0, 1 - x1, y1, 1 - x0]


def _crop_rot_cw(c):
    x0, y0, x1, y1 = c
    return [1 - y1, x0, 1 - y0, x1]


def _crop_flip(c):
    x0, y0, x1, y1 = c
    return [1 - x1, y0, 1 - x0, y1]


def transform_crop(crop, old_rot: int, old_flip: bool, new_rot: int, new_flip: bool):
    """Re-express a normalised crop after the 90-degree orientation changes."""
    if not crop:
        return crop
    c = list(crop)
    if old_flip:
        c = _crop_flip(c)
    for _ in range((old_rot // 90) % 4):
        c = _crop_rot_cw(c)
    for _ in range((new_rot // 90) % 4):
        c = _crop_rot_ccw(c)
    if new_flip:
        c = _crop_flip(c)
    return [round(v, 6) for v in c]


# Source <-> view coordinates. "Source" is the original negative as loaded (normalised
# 0..1); "view" is the oriented, straightened and (unless full_view) cropped picture, also
# normalised. Dust spots are stored in source coordinates so they stay on the dust when
# the frame is rotated, flipped, straightened or re-cropped.

def _view_frame(src_shape, s: Settings, full_view: bool):
    h, w = src_shape[:2]
    W, H = (h, w) if (int(s.rotation) // 90) % 2 else (w, h)
    c = [0.0, 0.0, 1.0, 1.0] if full_view or not s.crop else [min(max(float(v), 0.0), 1.0) for v in s.crop]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(s.straighten), 1.0)         if abs(float(s.straighten)) >= 1e-3 else None
    return w, h, W, H, c, M


def src_to_view(pts, src_shape, s: Settings, full_view: bool = False) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 2).copy()
    w, h, W, H, c, M = _view_frame(src_shape, s, full_view)
    x, y = p[:, 0] * w, p[:, 1] * h
    cw, ch = w, h
    for _ in range((int(s.rotation) // 90) % 4):        # np.rot90: counter-clockwise
        x, y = y, cw - x
        cw, ch = ch, cw
    if s.flip_h:
        x = W - x
    if M is not None:
        xi, yi = x - 0.5, y - 0.5
        x = M[0, 0] * xi + M[0, 1] * yi + M[0, 2] + 0.5
        y = M[1, 0] * xi + M[1, 1] * yi + M[1, 2] + 0.5
    vx = (x / W - c[0]) / max(c[2] - c[0], 1e-9)
    vy = (y / H - c[1]) / max(c[3] - c[1], 1e-9)
    return np.stack([vx, vy], axis=1)


def view_to_src(pts, src_shape, s: Settings, full_view: bool = False) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 2).copy()
    w, h, W, H, c, M = _view_frame(src_shape, s, full_view)
    x = (c[0] + p[:, 0] * (c[2] - c[0])) * W
    y = (c[1] + p[:, 1] * (c[3] - c[1])) * H
    if M is not None:
        Mi = cv2.invertAffineTransform(M)
        xi, yi = x - 0.5, y - 0.5
        x = Mi[0, 0] * xi + Mi[0, 1] * yi + Mi[0, 2] + 0.5
        y = Mi[1, 0] * xi + Mi[1, 1] * yi + Mi[1, 2] + 0.5
    if s.flip_h:
        x = W - x
    k = (int(s.rotation) // 90) % 4
    shapes = [(w, h)]
    for _ in range(k):
        shapes.append(shapes[-1][::-1])
    for i in range(k, 0, -1):                          # undo each rotation step
        pw, _ph = shapes[i - 1]
        x, y = pw - y, x
    return np.stack([x / w, y / h], axis=1)


def view_scale(src_shape, s: Settings, full_view: bool = False) -> float:
    """Long edge of the view divided by the long edge of the source (both in pixels)."""
    w, h, W, H, c, _ = _view_frame(src_shape, s, full_view)
    return max(W * (c[2] - c[0]), H * (c[3] - c[1])) / max(w, h)


# Crop aspect ratios (long side / short side) offered by the crop tool, and auto
# crop's guess of the format.
ASPECTS = {"3:2": 3 / 2, "4:3": 4 / 3, "5:4": 5 / 4, "1:1": 1.0, "6×7": 7 / 6, "16:9": 16 / 9,
           "65:24": 65 / 24}
DEFAULT_ASPECT = "3:2"       # 35 mm, the usual case when the format isn't recognised
ASPECT_TOLERANCE = 0.025     # how close (log ratio) a detected frame must be to another format


def crop_px(crop, frame_wh):
    W, H = frame_wh
    x0, y0, x1, y1 = crop or (0.0, 0.0, 1.0, 1.0)
    return x0 * W, y0 * H, x1 * W, y1 * H


def detect_aspect(crop, frame_wh) -> str:
    """Name of the standard format the crop is closest to, or DEFAULT_ASPECT."""
    x0, y0, x1, y1 = crop_px(crop, frame_wh)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    r = max(w, h) / min(w, h)
    name = min(ASPECTS, key=lambda k: abs(math.log(r / ASPECTS[k])))
    # Anything but 3:2 needs a close match: a frame detected a little too wide (clear
    # film at one end) must not turn into 16:9, or a slightly short one into 4:3.
    return name if abs(math.log(r / ASPECTS[name])) <= ASPECT_TOLERANCE else DEFAULT_ASPECT


def oriented_ratio(name: str, crop, frame_wh) -> float | None:
    """Pixel w/h for an aspect name, portrait or landscape to match the crop."""
    if name not in ASPECTS:
        return None
    x0, y0, x1, y1 = crop_px(crop, frame_wh)
    r = ASPECTS[name]
    return 1.0 / r if (y1 - y0) > (x1 - x0) else r


def fit_aspect(crop, frame_wh, ratio: float) -> list:
    """Largest crop with pixel aspect `ratio` (w/h) inside `crop`, same centre."""
    W, H = frame_wh
    x0, y0, x1, y1 = crop_px(crop, frame_wh)
    w, h = x1 - x0, y1 - y0
    if w / max(h, 1e-6) > ratio:
        w = h * ratio
    else:
        h = w / ratio
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [float(round((cx - w / 2) / W, 6)), float(round((cy - h / 2) / H, 6)),
            float(round((cx + w / 2) / W, 6)), float(round((cy + h / 2) / H, 6))]


def straighten_crop(crop, frame_wh, old_angle: float, new_angle: float):
    """Crop to use after changing the straighten angle from old to new.

    The picture turns about the crop centre (the crop stays where it is on the subject
    rather than the whole frame swinging round), and the crop shrinks, keeping its
    aspect, just enough that no empty corners come into view.
    """
    W, H = frame_wh
    if crop is None and abs(new_angle) < 1e-3:
        return None
    x0, y0, x1, y1 = crop_px(crop, frame_wh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
    m_old = cv2.getRotationMatrix2D((W / 2, H / 2), float(old_angle), 1.0)
    m_new = cv2.getRotationMatrix2D((W / 2, H / 2), float(new_angle), 1.0)
    px, py = cv2.invertAffineTransform(m_old) @ np.array([cx, cy, 1.0])
    cx, cy = m_new @ np.array([px, py, 1.0])
    inv = cv2.invertAffineTransform(m_new)
    eps = 1e-6

    def inside(x, y):
        if not (-eps <= x <= W + eps and -eps <= y <= H + eps):
            return False
        u, v = inv @ np.array([x, y, 1.0])
        return -eps <= u <= W + eps and -eps <= v <= H + eps

    if not inside(cx, cy):                       # centre fell off the picture: pull it back
        cx, cy = min(max(cx, 0.0), W), min(max(cy, 0.0), H)

    def fits(f):
        return all(inside(cx + sx * f * hw, cy + sy * f * hh) for sx in (-1, 1) for sy in (-1, 1))

    f = 1.0
    if not fits(1.0):
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fits(mid) else (lo, mid)
        f = lo
    hw, hh = hw * f, hh * f
    return [float(round(v, 6)) for v in ((cx - hw) / W, (cy - hh) / H, (cx + hw) / W, (cy + hh) / H)]


def resize_long_edge(img: np.ndarray, edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    if edge <= 0 or max(h, w) <= edge:
        return img
    scale = edge / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- analysis

def as_float(neg: np.ndarray) -> np.ndarray:
    """Negatives are float32 in [0, 1] or uint16 (full-resolution zoom cache)."""
    if neg.dtype == np.uint16:
        return neg.astype(np.float32) * np.float32(1.0 / 65535.0)
    return neg.astype(np.float32, copy=False)


def to_working(neg: np.ndarray, film_type: str) -> np.ndarray:
    """B&W film is processed as a single luminance channel."""
    if film_type == "bw":
        return (as_float(neg) @ LUMA)[..., None]
    return neg


def detect_film_type(neg_cropped: np.ndarray) -> str:
    """Guess "color" or "bw" from a (cropped) negative.

    A B&W image is one layer of silver, so pixel densities lie close to a single
    line in RGB density space; colour film has three independent dye layers that
    spread out. Its orange mask also gives a tell-tale low blue / high red base,
    which catches colour negatives of almost colourless scenes.
    """
    h, w = neg_cropped.shape[:2]
    s = min(1.0, 200.0 / max(h, w))
    small = cv2.resize(as_float(neg_cropped), (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA)
    px = small.reshape(-1, 3)
    d = density(px)
    lum = d.mean(axis=1)
    lo, hi = np.percentile(lum, [1, 99])
    keep = (lum > lo) & (lum < hi)
    if keep.sum() < 50:
        return "color"
    d = d[keep]
    ev = np.linalg.eigvalsh(np.cov((d - d.mean(axis=0)).T))
    spread = float(np.sqrt(max(ev[1], 0.0) / max(ev[2], 1e-12)))
    if spread >= 0.10:
        return "color"
    # Orange mask: blue < green < red in the clearest areas (nearest the film base).
    # Log ratios, so gamma-encoded scans (which squeeze ratios toward 1) still register.
    thin = px[keep][lum[keep] <= np.percentile(lum[keep], 5)]
    base = np.median(thin, axis=0) + 1e-6
    g_over_b, r_over_g = float(np.log(base[1] / base[2])), float(np.log(base[0] / base[1]))
    if spread >= 0.05 and g_over_b > 0.2 and r_over_g > 0.15:
        return "color"
    if g_over_b > 0.45 and r_over_g > 0.3:          # unmistakable mask, colourless scene
        return "color"
    return "bw"


def density(x: np.ndarray) -> np.ndarray:
    return -np.log10(np.maximum(x, EPS))


def analyze(neg: np.ndarray, s: Settings) -> dict:
    """Find per-channel density range (base / max) and grey-balance exponents."""
    w = to_working(neg, s.film_type)
    h, wd, nch = w.shape
    my, mx = int(h * 0.02), int(wd * 0.02)
    sub = w[my:h - my, mx:wd - mx] if h > 50 and wd > 50 else w
    step = max(1, int(math.sqrt(sub.shape[0] * sub.shape[1] / 120_000)))
    d = density(as_float(sub[::step, ::step]).reshape(-1, nch))

    # Per-channel levels remove the orange mask and balance the channels; they always
    # use a tiny fixed clip. The user's clip sliders act afterwards on brightness only.
    if s.base_rgb:
        base = to_working(np.asarray(s.base_rgb, np.float32).reshape(1, 1, 3), s.film_type)
        dmin = density(base.reshape(-1))
    else:
        dmin = np.percentile(d, AUTO_CLIP, axis=0)
    dmax = np.percentile(d, 100.0 - AUTO_CLIP, axis=0)
    dmax = np.maximum(dmax, dmin + 0.05)

    wb_gamma = np.ones(nch)
    if s.auto_wb and nch == 3:
        t = (d - dmin) / (dmax - dmin)
        t = t[np.all((t > 0.06) & (t < 0.94), axis=1)]
        if len(t) > 500:
            # Grey-pixel balance: repeatedly pick the most neutral-looking pixels
            # under the current balance and equalise their channel medians.
            # Much less fooled by big coloured areas (sky, foliage) than grey-world.
            for _ in range(4):
                tb = _balance(t.copy(), wb_gamma)
                chroma = tb.max(axis=1) - tb.min(axis=1)
                sel = chroma < 0.08
                if sel.sum() < max(300, 0.03 * len(t)):
                    sel = chroma <= np.percentile(chroma, 5)
                med = np.clip(np.median(t[sel], axis=0), 0.02, 0.98)
                target = float(np.exp(np.mean(np.log(med))))
                wb_gamma = np.clip(np.log(target) / np.log(med), 0.5, 2.0)

    gain = _auto_gain(d, dmin, dmax, wb_gamma) if s.auto_brightness else 1.0
    v_black, v_white = _clip_points(d, dmin, dmax, wb_gamma, gain, s.clip_black, s.clip_white)
    return {
        "dmin": [float(v) for v in dmin],
        "dmax": [float(v) for v in dmax],
        "wb_gamma": [float(v) for v in wb_gamma],
        "auto_gain": gain,
        "v_black": v_black,
        "v_white": v_white,
        "key": analysis_key(s),
    }


AUTO_CLIP = 0.05       # % per channel that the automatic levels ignore (dust, specular glints)


def _brightness_t(d, dmin, dmax, wb_gamma) -> np.ndarray:
    """Normalised, grey-balanced density of each sample."""
    t = np.clip((d - dmin) / (dmax - dmin), 0.0, 1.5).astype(np.float32)
    return _balance(t, wb_gamma)


def _clip_points(d, dmin, dmax, wb_gamma, gain, clip_black, clip_white) -> tuple[float, float]:
    """Display-level black / white points that turn the requested share of the picture
    pure black / white. They apply after auto levels and auto brightness and before the
    tone sliders, and move all channels together, so colours don't shift."""
    cb, cw = float(clip_black or 0.0), float(clip_white or 0.0)
    if cb <= 0.0 and cw <= 0.0:
        return 0.0, 1.0
    t = _brightness_t(d, dmin, dmax, wb_gamma)
    v = np.maximum(_t_to_linear(t, _stops(t.shape[1])), 0.0)
    if gain > 1.0001:
        _lift(v, gain)
    v = np.minimum(v, 1.0) ** (1.0 / DISPLAY_GAMMA)
    y = v @ LUMA if v.shape[1] == 3 else v[:, 0]
    vb = float(np.percentile(y, min(cb, 50.0))) if cb > 0 else 0.0
    vw = float(np.percentile(y, 100.0 - min(cw, 50.0))) if cw > 0 else 1.0
    if vw - vb < 0.05:
        vw = vb + 0.05
    return round(vb, 5), round(vw, 5)


AUTO_WHITE = 0.79      # linear level (display ~0.90) the brightest real content is lifted to
AUTO_MID_MAX = 0.22    # linear level (display ~0.50) the median is never lifted above
AUTO_GAIN_MAX = (2.5, 4.0)   # ~1.3 stops for night / low-key frames, 2 stops otherwise


def _auto_gain(d: np.ndarray, dmin, dmax, wb_gamma) -> float:
    """Exposure lift for conversions that came out dim.

    The white point of auto levels often sits on a few specular pixels (lamps, reflections),
    which leaves the actual picture dull and grey. This picks a gain that brings the
    brightest real content (99th percentile) up toward white, without pushing the median
    past a normal brightness. Mostly dark (night, low-key) frames get a smaller lift, so they
    stay dark. It is applied by _lift: a plain gain near black (blacks stay
    black, no colour shifts, night stays night) that rolls off so nothing clips.
    """
    t = _brightness_t(d, dmin, dmax, wb_gamma)
    lin = np.maximum(_t_to_linear(t, _stops(t.shape[1])), 0.0)
    y = lin @ LUMA if lin.shape[1] == 3 else lin[:, 0]
    med, hi = (float(x) for x in np.percentile(y, [50, 99]))

    def gain_for(x, target):          # k such that _lift(x, k) == target
        x = min(max(x, 1e-4), 0.999)
        return target * (1.0 - x) / (x * (1.0 - target))

    lo_key, normal = AUTO_GAIN_MAX
    cap = float(np.interp(med ** (1.0 / DISPLAY_GAMMA), (0.15, 0.30), (lo_key, normal)))
    k = min(gain_for(hi, AUTO_WHITE), gain_for(med, AUTO_MID_MAX), cap)
    return float(max(k, 1.0))


def _lift(v: np.ndarray, k: float) -> np.ndarray:
    """Gain k on linear light with a highlight roll-off: 0 -> 0, 1 -> 1, slope k at black."""
    v *= k
    v /= 1.0 + (k - 1.0) / k * v
    return v


def analysis_valid(a: dict | None, s: Settings) -> bool:
    if not a or "dmin" not in a:
        return False
    nch = 1 if s.film_type == "bw" else 3
    if len(a["dmin"]) != nch:
        return False
    if not s.lock_analysis and ("v_black" not in a or (s.auto_brightness and "auto_gain" not in a)):
        return False                    # saved by an older version
    return bool(s.lock_analysis or a.get("key") == analysis_key(s))


def resolve_analysis(neg_cropped: np.ndarray, s: Settings) -> dict:
    return s.analysis if analysis_valid(s.analysis, s) else analyze(neg_cropped, s)


# --------------------------------------------------------------------------- rendering

def _gains(s: Settings, nch: int) -> np.ndarray:
    ev = 2.0 ** float(s.exposure)
    if nch == 1:
        return np.array([ev], np.float32)
    tk, tn = s.temperature / 100.0, s.tint / 100.0
    c = np.array([2 ** (0.5 * tk), 2 ** (-0.4 * tn), 2 ** (-0.5 * tk)], np.float64)
    c *= np.asarray(s.wb_gains, np.float64)
    c /= float(c @ LUMA.astype(np.float64))
    return (c * ev).astype(np.float32)


def _to_t(block: np.ndarray, a: dict) -> np.ndarray:
    dmin = np.asarray(a["dmin"], np.float32)
    dmax = np.asarray(a["dmax"], np.float32)
    wg = np.asarray(a.get("wb_gamma", [1.0] * len(dmin)), np.float32)
    t = density(block).astype(np.float32, copy=False)
    t -= dmin
    t /= (dmax - dmin)
    np.clip(t, 0.0, 1.5, out=t)
    return _balance(t, wg)


WB_TOE = 0.2


def _balance(t: np.ndarray, wb_gamma) -> np.ndarray:
    """Per-channel grey balance t ** g, eased into a neutral toe below WB_TOE.

    A pure power has infinite (g < 1) or zero (g > 1) slope at black, which tints the
    deepest shadows. Below WB_TOE a cubic takes over that starts at slope 1 for every
    channel (neutral blacks) and meets the power curve in value and slope.
    """
    g = np.asarray(wb_gamma, np.float32)
    if not np.any(np.abs(g - 1.0) > 1e-4):
        return t
    t0 = np.float32(WB_TOE)
    e = t0 ** g - t0                         # value and slope offsets at the joint
    f = g * t0 ** (g - 1.0) - 1.0
    a = (3.0 * e - f * t0) / t0 ** 2
    b = (f * t0 - 2.0 * e) / t0 ** 3
    lo = t * (1.0 + t * (a + t * b))
    toe = t < t0
    t **= g
    np.copyto(t, np.maximum(lo, 0.0), where=toe)
    return t


def _stops(nch: int) -> float:
    return PRINT_STOPS_BW if nch == 1 else PRINT_STOPS


def _t_to_linear(t: np.ndarray, stops: float = PRINT_STOPS) -> np.ndarray:
    """Normalised density -> scene-linear light, like printing onto paper.

    Density is proportional to log exposure, so exposure is exponential in t.
    PRINT_STOPS is the scene range spanned between the black and white points;
    it puts an 18% grey card at ~18% linear for typical negatives.
    """
    t *= stops
    np.exp2(t, out=t)
    t -= 1.0
    t /= 2.0 ** stops - 1.0
    return t


LUT_SIZE = 65536


def _curves(s: Settings, a: dict, nch: int) -> np.ndarray:
    """Every per-channel step as one lookup table: negative value (0..65535) -> output."""
    x = np.arange(LUT_SIZE, dtype=np.float32) / (LUT_SIZE - 1)
    v = _t_to_linear(_to_t(np.repeat(x[:, None], nch, axis=1), a), _stops(nch))
    gains = _gains(s, nch)
    if not np.allclose(gains, 1.0, atol=1e-4):
        v *= gains
    np.maximum(v, 0.0, out=v)
    k = float(a.get("auto_gain", 1.0)) if s.auto_brightness else 1.0
    if k > 1.0001:
        _lift(v, k)
    v **= 1.0 / DISPLAY_GAMMA
    vb, vw = float(a.get("v_black", 0.0)), float(a.get("v_white", 1.0))
    if vb != 0.0 or vw != 1.0:                          # Clip to black / white sliders
        v -= vb
        v /= max(vw - vb, 0.05)

    b, w = -s.black / 200.0, 1.0 - s.white / 200.0     # Blacks + lifts, - deepens
    if b != 0.0 or w != 1.0:
        v -= b
        v /= max(w - b, 0.05)
    np.clip(v, 0.0, 1.0, out=v)

    if abs(s.gamma - 1.0) > 1e-4:
        v **= 1.0 / max(s.gamma, 0.05)
    if s.shadows:
        k = s.shadows / 100.0
        k = 1.5 * k if k > 0 else k
        v += k * v * (1.0 - v) ** 2
    if s.highlights:
        k = s.highlights / 100.0
        k = k if k > 0 else 1.5 * k
        v += k * v * v * (1.0 - v)
    if s.contrast:
        _contrast(v, s.contrast / 100.0)
    np.clip(v, 0.0, 1.0, out=v)
    return np.ascontiguousarray(v.T)


CONTRAST_STEEPNESS = 12.0   # sigmoid steepness at contrast +100 (mid-tone slope ~3x)


def _contrast(v: np.ndarray, k: float) -> np.ndarray:
    """S-curve around mid-grey, in place. 0 and 1 stay fixed.

    k > 0: a normalised logistic curve, mid-tone slope ~1 + a^2/48 (a = steepness).
    k < 0: its exact inverse, which flattens the mid-tones by the same amount.
    """
    a = CONTRAST_STEEPNESS * abs(k)
    if a < 1e-3:
        return v
    lo = 1.0 / (1.0 + math.exp(a / 2.0))
    span = 1.0 - 2.0 * lo
    np.clip(v, 0.0, 1.0, out=v)
    if k > 0:
        v -= 0.5
        v *= -a
        np.exp(v, out=v)
        v += 1.0
        np.reciprocal(v, out=v)
        v -= lo
        v /= span
    else:
        v *= span
        v += lo
        np.clip(v, 1e-7, 1.0 - 1e-7, out=v)
        v[...] = 0.5 + np.log(v / (1.0 - v)) / a
    return v


def render(neg_cropped: np.ndarray, s: Settings, a: dict, ref_size: int | None = None) -> np.ndarray:
    """Render a (geometry-applied) negative into a display-referred sRGB float image.

    ref_size is the long edge of the whole picture when rendering only part of it
    (zoomed view), so sharpening matches the full render.
    """
    w = to_working(neg_cropped, s.film_type)
    h, nch = w.shape[0], w.shape[2]
    direct = w.dtype == np.uint16          # uint16 values are LUT indices already
    luts = _curves(s, a, nch)
    out = np.empty(w.shape[:2] + (3,), np.float32)
    sat = 1.0 + s.saturation / 100.0
    sat_m = ((1.0 - sat) * np.tile(LUMA, (3, 1)) + sat * np.eye(3)).astype(np.float32)
    for y in range(0, h, 512):
        if direct:
            qi = w[y:y + 512]
        else:
            q = w[y:y + 512] * float(LUT_SIZE - 1)
            q += 0.5
            np.clip(q, 0, LUT_SIZE - 1, out=q)
            qi = q.astype(np.uint16)
        blk = out[y:y + 512]
        if nch == 1:
            blk[...] = luts[0][qi[..., 0]][..., None]
            continue
        for c in range(3):
            blk[..., c] = luts[c][qi[..., c]]
        if s.saturation:
            blk[...] = cv2.transform(blk, sat_m)
            np.clip(blk, 0.0, 1.0, out=blk)
    if s.denoise > 0:
        _denoise(out, s.denoise / 100.0, ref_size or max(out.shape[:2]), mono=nch == 1)
    if s.sharpen > 0:
        sigma = max(0.5, (ref_size or max(out.shape[:2])) / 4000.0)
        blur = cv2.GaussianBlur(out, (0, 0), sigma)
        amount = s.sharpen / 100.0 * 1.5
        out += amount * (out - blur)
        del blur
        np.clip(out, 0.0, 1.0, out=out)
    return out


DENOISE_SCALE = 1.2     # filter sigma per 4000 px of the picture's long edge


def _denoise(img: np.ndarray, k: float, ref_size: int, mono: bool) -> None:
    """Gentle reduction of digital scan noise, in place.

    Colour speckle (chroma noise from the camera / scanner sensor, worst in the dense
    parts of the negative) is blurred away; fine luminance noise is smoothed only where
    it is a small deviation from its surroundings, so edges, detail and most film grain
    survive. Sized relative to the picture, so preview, zoom and export agree; at preview
    size the noise is already averaged out and nothing needs doing.
    """
    sigma = DENOISE_SCALE * ref_size / 4000.0 * (0.6 + 0.8 * k)
    if sigma < 0.35:
        return
    thr = 0.012 + 0.03 * k                         # luminance deviation still treated as noise
    chroma_amt = min(1.0, 1.6 * k)
    pad = int(math.ceil(sigma * 2.5 * 3)) + 2
    h = img.shape[0]
    step = 768
    for y0 in range(0, h, step):
        a, b = max(0, y0 - pad), min(h, y0 + step + pad)
        blk = img[a:b]
        lum = blk @ LUMA if not mono else blk[..., 0].copy()
        dev = lum - cv2.GaussianBlur(lum, (0, 0), sigma)
        wgt = np.exp(-np.square(dev / thr)) * k
        new_l = lum - wgt * dev
        if mono:
            res = np.repeat(new_l[..., None], 3, axis=2)
        else:
            chroma = blk - lum[..., None]
            soft = cv2.GaussianBlur(chroma, (0, 0), sigma * 2.5)
            res = new_l[..., None] + chroma + chroma_amt * (soft - chroma)
        o0 = y0 - a
        img[y0:min(h, y0 + step)] = res[o0:o0 + min(step, h - y0)]
    np.clip(img, 0.0, 1.0, out=img)


def neutral_wb_gains(sample_rgb, s: Settings, a: dict) -> list:
    """White-balance gains that make a sampled (negative) colour neutral."""
    t = _to_t(np.asarray(sample_rgb, np.float32).reshape(1, 1, 3), a)
    lin = np.maximum(_t_to_linear(t).reshape(3).astype(np.float64), 1e-4)
    g = lin.mean() / lin
    return [float(x) for x in g / g[1]]


def negative_preview(neg: np.ndarray, scale: float | None = None) -> np.ndarray:
    neg = as_float(neg)
    if scale is None:
        scale = negative_scale(neg)
    return np.clip(neg / scale, 0.0, 1.0) ** (1.0 / DISPLAY_GAMMA)


def negative_scale(neg: np.ndarray) -> float:
    return float(np.percentile(as_float(neg[::4, ::4]), 99.5)) or 1.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((img * 255.0 + 0.5).astype(np.uint8))


def to_uint16(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((img * 65535.0 + 0.5).astype(np.uint16))


WAVE_COLS, WAVE_LEVELS = 256, 128


def waveform(img8: np.ndarray, mono: bool = False) -> np.ndarray:
    """Waveform monitor data: for each of WAVE_COLS image columns, how many pixels sit
    at each of WAVE_LEVELS brightness levels; per channel (R, G, B) or luminance only.
    Shape (channels, WAVE_LEVELS, WAVE_COLS), level 0 = black."""
    h, w = img8.shape[:2]
    step = max(1, int(math.sqrt(h * w / 300_000)))
    sub = img8[::step, ::step]
    sh, sw = sub.shape[:2]
    col = (np.arange(sw) * WAVE_COLS // max(sw, 1)).astype(np.int64)
    col = np.broadcast_to(col, (sh, sw)).ravel()
    planes = [(sub.astype(np.float32) @ LUMA).astype(np.uint8)] if mono else [sub[..., c] for c in range(3)]
    out = np.empty((len(planes), WAVE_LEVELS, WAVE_COLS), np.float32)
    for i, plane in enumerate(planes):
        lvl = (plane.ravel().astype(np.int64) * WAVE_LEVELS) >> 8
        out[i] = np.bincount(lvl * WAVE_COLS + col, minlength=WAVE_LEVELS * WAVE_COLS) \
            .reshape(WAVE_LEVELS, WAVE_COLS)
    return out


def histogram(img8: np.ndarray) -> np.ndarray:
    step = max(1, int(math.sqrt(img8.shape[0] * img8.shape[1] / 200_000)))
    sub = img8[::step, ::step].reshape(-1, 3)
    return np.stack([np.bincount(sub[:, c], minlength=256) for c in range(3)]).astype(np.float32)


def sample(img: np.ndarray, nx: float, ny: float, radius: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    x = int(min(max(nx, 0.0), 1.0) * (w - 1))
    y = int(min(max(ny, 0.0), 1.0) * (h - 1))
    patch = img[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    return np.median(patch.reshape(-1, img.shape[2]), axis=0)
