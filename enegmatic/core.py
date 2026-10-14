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
_PRINT_SCALE = 2.0 ** PRINT_STOPS - 1.0
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
    "detail": ("Detail", ("sharpen",)),
    "orientation": ("Rotation & flip", ("rotation", "flip_h")),
    "crop": ("Crop & straighten", ("crop", "straighten")),
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
    # inversion / analysis
    clip_black: float = 0.05            # % of pixels auto levels may clip to black
    clip_white: float = 0.05            # % of pixels auto levels may clip to white
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
    contrast: float = 15.0
    gamma: float = 1.0
    black: float = 0.0
    white: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    sharpen: float = 0.0

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

    if s.base_rgb:
        base = to_working(np.asarray(s.base_rgb, np.float32).reshape(1, 1, 3), s.film_type)
        dmin = density(base.reshape(-1))
    else:
        dmin = np.percentile(d, float(s.clip_black), axis=0)
    dmax = np.percentile(d, 100.0 - float(s.clip_white), axis=0)
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
                tb = t ** wb_gamma
                chroma = tb.max(axis=1) - tb.min(axis=1)
                sel = chroma < 0.08
                if sel.sum() < max(300, 0.03 * len(t)):
                    sel = chroma <= np.percentile(chroma, 5)
                med = np.clip(np.median(t[sel], axis=0), 0.02, 0.98)
                target = float(np.exp(np.mean(np.log(med))))
                wb_gamma = np.clip(np.log(target) / np.log(med), 0.5, 2.0)

    return {
        "dmin": [float(v) for v in dmin],
        "dmax": [float(v) for v in dmax],
        "wb_gamma": [float(v) for v in wb_gamma],
        "auto_gamma": _auto_gamma(d, dmin, dmax, wb_gamma) if s.auto_brightness else 1.0,
        "key": analysis_key(s),
    }


BRIGHTNESS_TARGET = 0.42   # median display brightness of a normally exposed photo


def _auto_gamma(d: np.ndarray, dmin, dmax, wb_gamma) -> float:
    """Midtone lift that brings the conversion's median brightness toward a normal photo.

    Applied as a gamma on the display values, so black and white points (and therefore
    clipping) stay where auto levels put them; only midtones and shadows move. Frames that
    are mostly genuinely dark (night, low-key) keep their darkness, and bright frames are
    never darkened.
    """
    t = np.clip((d - dmin) / (dmax - dmin), 0.0, 1.5).astype(np.float32)
    t **= np.asarray(wb_gamma, np.float32)
    v = np.maximum(_t_to_linear(t), 0.0) ** (1.0 / DISPLAY_GAMMA)
    y = v @ LUMA if v.shape[1] == 3 else v[:, 0]
    med, p75 = (float(x) for x in np.percentile(y, [50, 75]))
    if med >= BRIGHTNESS_TARGET:
        return 1.0
    g = np.log(max(med, 0.02)) / np.log(BRIGHTNESS_TARGET)
    # Low-key protection: when three quarters of the frame is near black (night scenes,
    # dark backgrounds) the darkness is the picture, so fade the lift out.
    strength = 0.8 * float(np.clip((p75 - 0.12) / 0.12, 0.0, 1.0))
    return float(min(1.0 + (g - 1.0) * strength, 1.8))


def analysis_valid(a: dict | None, s: Settings) -> bool:
    if not a or "dmin" not in a:
        return False
    nch = 1 if s.film_type == "bw" else 3
    if len(a["dmin"]) != nch:
        return False
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
    if np.any(np.abs(wg - 1.0) > 1e-4):
        t **= wg
    return t


def _t_to_linear(t: np.ndarray) -> np.ndarray:
    """Normalised density -> scene-linear light, like printing onto paper.

    Density is proportional to log exposure, so exposure is exponential in t.
    PRINT_STOPS is the scene range spanned between the black and white points;
    it puts an 18% grey card at ~18% linear for typical negatives.
    """
    t *= PRINT_STOPS
    np.exp2(t, out=t)
    t -= 1.0
    t /= _PRINT_SCALE
    return t


LUT_SIZE = 65536


def _curves(s: Settings, a: dict, nch: int) -> np.ndarray:
    """Every per-channel step as one lookup table: negative value (0..65535) -> output."""
    x = np.arange(LUT_SIZE, dtype=np.float32) / (LUT_SIZE - 1)
    v = _t_to_linear(_to_t(np.repeat(x[:, None], nch, axis=1), a))
    gains = _gains(s, nch)
    if not np.allclose(gains, 1.0, atol=1e-4):
        v *= gains
    np.maximum(v, 0.0, out=v)
    v **= 1.0 / DISPLAY_GAMMA
    ag = float(a.get("auto_gamma", 1.0)) if s.auto_brightness else 1.0
    if abs(ag - 1.0) > 1e-4:
        v **= 1.0 / ag

    b, w = s.black / 200.0, 1.0 - s.white / 200.0
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
        k = s.contrast / 100.0
        v += k * (v * v * (3.0 - 2.0 * v) - v)
    np.clip(v, 0.0, 1.0, out=v)
    return np.ascontiguousarray(v.T)


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
    if s.sharpen > 0:
        sigma = max(0.5, (ref_size or max(out.shape[:2])) / 4000.0)
        blur = cv2.GaussianBlur(out, (0, 0), sigma)
        amount = s.sharpen / 100.0 * 1.5
        out += amount * (out - blur)
        del blur
        np.clip(out, 0.0, 1.0, out=out)
    return out


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
