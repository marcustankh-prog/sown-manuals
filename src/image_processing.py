"""Deterministic image cleanup pipeline for user-supplied step photos and
sketches. No generative AI — pure Pillow + rembg operations.

Public API:
    detect_kind(path) -> "sketch" | "photo"
    clean_image(src, dest, *, mode="auto", bg="remove",
                crop_box=None, size=1200) -> Path

`mode`:
    "auto"     — run detect_kind() and pick line_art for sketches, photo otherwise
    "line_art" — convert to crisp black-on-white outlines
    "photo"    — keep colour, just clean up the background and edges

`bg`:
    "remove"   — alpha-cut the background (transparent PNG)
    "keep"     — leave the original background intact (paper white for sketches)

`crop_box` is an optional (left, top, right, bottom) tuple in source-pixel
coordinates applied before any other processing. None = auto crop to content.

The output is always a square PNG of side `size`.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Tuple

from PIL import Image, ImageOps, ImageFilter, ImageStat, ImageDraw

Mode = Literal["auto", "line_art", "photo"]
Bg = Literal["remove", "keep"]


def _apply_rounded_corners(img: Image.Image, radius: int) -> Image.Image:
    """Return an RGBA image with the given corner radius applied to alpha.

    Preserves any existing transparency by intersecting with a rounded mask.
    """
    if radius <= 0:
        return img
    from PIL import ImageChops
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, img.width, img.height), radius=radius, fill=255,
    )
    existing_alpha = img.split()[3]
    img.putalpha(ImageChops.multiply(existing_alpha, mask))
    return img


# ---- Detection ------------------------------------------------------------

def detect_kind(path: str | Path) -> str:
    """Heuristic: hand-drawn sketches have low colour saturation, a strong
    near-white background mode in the histogram, and bimodal luminance.
    Photos do not. Returns "sketch" or "photo"."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((512, 512))

    # Mean saturation
    hsv = img.convert("HSV")
    sat_mean = ImageStat.Stat(hsv).mean[1]  # 0..255

    # Fraction of near-white pixels
    gray = img.convert("L")
    hist = gray.histogram()
    near_white = sum(hist[235:]) / max(sum(hist), 1)

    # Sketch if low colour AND lots of paper-white pixels
    if sat_mean < 35 and near_white > 0.35:
        return "sketch"
    return "photo"


# ---- Background removal --------------------------------------------------

_REMBG_SESSION = None


def _remove_bg(img: Image.Image) -> Image.Image:
    """Alpha-cut the background using rembg (lazy-loaded ONNX model)."""
    global _REMBG_SESSION
    try:
        from rembg import remove, new_session  # lazy heavy import
    except Exception as exc:  # noqa: BLE001
        # Fall back: keep image opaque if rembg isn't installed.
        raise RuntimeError(
            "rembg is required for background removal. "
            "Install with `pip install rembg`."
        ) from exc

    if _REMBG_SESSION is None:
        # u2netp is small (~4 MB) vs u2net (~170 MB) — good default for slugs.
        _REMBG_SESSION = new_session("u2netp")

    buf = BytesIO()
    img.save(buf, format="PNG")
    out = remove(buf.getvalue(), session=_REMBG_SESSION)
    return Image.open(BytesIO(out)).convert("RGBA")


# ---- Auto-crop, pad to square -------------------------------------------

def _auto_crop_to_content(img: Image.Image, pad_pct: float = 0.08) -> Image.Image:
    """Crop tightly around the visible (non-transparent / non-white) content,
    then add a percentage padding around it."""
    # Build an alpha-or-darkness mask
    if img.mode == "RGBA":
        mask = img.split()[3]
    else:
        gray = img.convert("L")
        # treat anything darker than ~245 as content
        mask = gray.point(lambda v: 255 if v < 245 else 0)

    bbox = mask.getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    w, h = right - left, bottom - top
    pad = int(max(w, h) * pad_pct)
    L = max(0, left - pad)
    T = max(0, top - pad)
    R = min(img.width, right + pad)
    B = min(img.height, bottom + pad)
    return img.crop((L, T, R, B))


def _pad_to_square(img: Image.Image, fill=(0, 0, 0, 0)) -> Image.Image:
    """Centre on a transparent (or paper-white) square canvas."""
    side = max(img.width, img.height)
    if img.mode == "RGBA":
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    else:
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2),
                 img if img.mode == "RGBA" else None)
    return canvas


# ---- Style passes --------------------------------------------------------

def _to_line_art(img: Image.Image) -> Image.Image:
    """Convert to crisp black ink on white. Drops all colour/grey midtones."""
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    gray = rgb.convert("L")
    # Mild blur first to suppress paper grain, then unsharp to sharpen edges.
    gray = gray.filter(ImageFilter.GaussianBlur(radius=0.6))
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.2, percent=180))
    # Adaptive-ish threshold: anchor on the histogram's brightness mode.
    hist = gray.histogram()
    bright_mode = max(range(180, 256), key=lambda i: hist[i])
    threshold = max(110, bright_mode - 60)
    bw = gray.point(lambda v: 0 if v < threshold else 255, mode="L")
    return bw.convert("RGB")  # final pass keeps it as RGB white bg


def _enhance_photo(img: Image.Image) -> Image.Image:
    """Gentle photo cleanup: auto-contrast + unsharp + denoise."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    rgb = img.convert("RGB")
    rgb = ImageOps.autocontrast(rgb, cutoff=1)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120))
    if img.mode == "RGBA":
        # preserve alpha
        out = Image.merge("RGBA", (*rgb.split(), img.split()[3]))
        return out
    return rgb


# ---- Public entry point --------------------------------------------------

def clean_image(
    src: str | Path,
    dest: str | Path,
    *,
    mode: Mode = "auto",
    bg: Bg = "remove",
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    size: int = 1200,
    corner_radius: Optional[int] = None,
) -> Path:
    """Clean an image and write it to `dest` as a square PNG.

    `corner_radius` is the rounded-corner radius in output pixels. When
    `None` (default), it's set to ~6% of `size`. Pass `0` to disable.
    """
    src_p = Path(src)
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src_p)
    img = ImageOps.exif_transpose(img)  # rotate per camera orientation

    if crop_box:
        img = img.crop(crop_box)

    # Pick mode
    chosen = mode
    if mode == "auto":
        chosen = "line_art" if detect_kind(src_p) == "sketch" else "photo"

    # Background pass
    if bg == "remove":
        try:
            img = _remove_bg(img.convert("RGB"))
        except RuntimeError:
            img = img.convert("RGBA")
    else:
        img = img.convert("RGB")

    # Style pass
    if chosen == "line_art":
        # Line art is always presented on white paper, regardless of bg.
        styled = _to_line_art(img)
    else:
        styled = _enhance_photo(img)

    # Crop and square
    cropped = _auto_crop_to_content(styled)
    squared = _pad_to_square(cropped)

    # Resize to target side
    squared = squared.resize((size, size), Image.LANCZOS)

    # Rounded corners (default ~6% of side).
    radius = corner_radius if corner_radius is not None else round(size * 0.06)
    squared = _apply_rounded_corners(squared, radius)

    squared.save(dest_p, format="PNG", optimize=True)
    return dest_p
