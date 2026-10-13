"""Full-resolution export. Qt-free so it can run in worker processes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .core import Settings, apply_crop, apply_geometry, render, resize_long_edge, resolve_analysis
from .fileio import FORMATS, load_negative, save_image


@dataclass
class ExportOptions:
    fmt: str = "jpeg"           # jpeg | tiff | png
    bits: int = 8               # 8 | 16 (tiff / png)
    quality: int = 95
    long_edge: int = 0          # 0 = full resolution
    use_subfolder: bool = True  # write into <source folder>/<subfolder>
    subfolder: str = "positives"
    out_dir: str = ""
    suffix: str = ""
    overwrite: bool = False
    bw_grayscale: bool = True   # B&W film -> single-channel file
    workers: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ExportOptions":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


def output_path(src: str, o: ExportOptions) -> Path:
    p = Path(src)
    folder = p.parent / (o.subfolder or "positives") if o.use_subfolder else Path(o.out_dir or p.parent)
    folder.mkdir(parents=True, exist_ok=True)
    ext = FORMATS[o.fmt][0]
    out = folder / f"{p.stem}{o.suffix}{ext}"
    i = 1
    while not o.overwrite and out.exists():
        out = folder / f"{p.stem}{o.suffix}_{i}{ext}"
        i += 1
    return out


def export_one(src: str, settings: dict, options: dict) -> tuple[str, str | None, str | None]:
    """Returns (source, output path or None, error or None)."""
    try:
        s = Settings.from_dict(settings)
        o = ExportOptions.from_dict(options)
        neg = load_negative(src, half_size=False)
        full = apply_geometry(neg, s, crop=False)
        del neg
        view = apply_crop(full, s.crop)
        a = resolve_analysis(view, s)
        img = render(view, s, a)
        del full, view
        img = resize_long_edge(img, int(o.long_edge))
        out = output_path(src, o)
        save_image(str(out), img, o.fmt, int(o.bits), int(o.quality),
                   grayscale=o.bw_grayscale and s.film_type == "bw")
        return src, str(out), None
    except Exception as e:  # reported back to the UI
        return src, None, f"{type(e).__name__}: {e}"
