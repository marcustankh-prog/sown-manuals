"""Render a Flower into HTML and PDF.

PDF backend: WeasyPrint. On Windows, GTK3 runtime must be installed; if it
isn't, calling `to_pdf` will raise with a helpful message.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import sys
from pathlib import Path
from typing import Tuple
from urllib.parse import unquote, urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import Flower

ROOT = Path(__file__).parent
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"


def _to_data_uri(src: str) -> str:
    """Convert a file:// URI or local path to a base64 data: URI.

    Browsers (including Chromium under Playwright and Streamlit's iframe)
    block file:// subresources, so we inline images for portable rendering.
    """
    if not src or src.startswith(("data:", "http://", "https://")):
        return src
    if src.startswith("file:"):
        raw = unquote(urlparse(src).path)
        # On Windows, file URIs look like file:///C:/Users/... so urlparse
        # returns "/C:/Users/..." — drop the leading slash. On POSIX the
        # leading slash IS the absolute root and must be preserved.
        if os.name == "nt" and len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        path = Path(raw)
    else:
        path = Path(src)
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(
            f"[pdf_generator] could not inline image {src!r} (resolved to {path}): {exc}",
            file=sys.stderr,
        )
        return src
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _inline_flower_images(flower: Flower) -> Flower:
    """Return a copy of `flower` with every image src as a data URI."""
    f = flower.model_copy(deep=True)
    if f.hero_image:
        f.hero_image = _to_data_uri(f.hero_image)
    diagram = getattr(f, "anatomy_diagram", None)
    if diagram:
        diagram.path = _to_data_uri(diagram.path)
    for comp in f.components:
        for img in comp.images:
            img.path = _to_data_uri(img.path)
    if f.assembly:
        for img in f.assembly.images:
            img.path = _to_data_uri(img.path)
    for img in f.inspo_images:
        img.path = _to_data_uri(img.path)
    return f


def _hex_lighten(hex_color: str, amount: float = 0.85) -> str:
    """Return a softer pastel version of a hex color (mix with white)."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return "#eef0e6"
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return 1.0
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _accent_for(flower: Flower) -> Tuple[str, str]:
    # Skip near-white palette entries — they render as invisible headings on
    # the cream paper background. Fall back to the next dark-enough swatch,
    # or the default sage if nothing in the palette is readable.
    accent = next(
        (c for c in (flower.palette or []) if _luminance(c) < 0.7),
        "#8A9180",
    )
    return accent, _hex_lighten(accent, 0.85)


_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=False,
    lstrip_blocks=False,
)


def render_html(flower: Flower, *, embed_css: bool = True) -> str:
    flower = _inline_flower_images(flower)
    accent, accent_soft = _accent_for(flower)
    template = _env.get_template("manual.html.j2")
    if embed_css:
        css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
        # Inline the stylesheet via <style> rather than linking it.
        html = template.render(
            flower=flower,
            stylesheet_href="about:blank",
            accent=accent,
            accent_soft=accent_soft,
        )
        # Inject inline <style> right before </head>
        html = html.replace(
            '<link rel="stylesheet" href="about:blank">',
            f"<style>{css}</style>",
        )
        return html
    return template.render(
        flower=flower,
        stylesheet_href=str((STATIC_DIR / "style.css").resolve().as_uri()),
        accent=accent,
        accent_soft=accent_soft,
    )


def to_pdf(flower: Flower, output_path: Path, *, backend: str = "auto") -> Path:
    """Render the flower manual to a PDF at output_path.

    backend:
      - "auto"        try WeasyPrint first, fall back to Playwright/Chromium.
      - "weasyprint"  WeasyPrint only (needs GTK3 on Windows).
      - "playwright"  Chromium only (run `playwright install chromium` once).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_str = render_html(flower, embed_css=True)

    errors: list[str] = []

    if backend in ("auto", "weasyprint"):
        try:
            from weasyprint import HTML  # type: ignore

            HTML(string=html_str, base_url=str(ROOT)).write_pdf(target=str(output_path))
            return output_path
        except (OSError, ImportError) as e:
            errors.append(f"weasyprint: {e}")
            if backend == "weasyprint":
                raise RuntimeError(
                    "WeasyPrint failed. On Windows install the GTK3 runtime "
                    "(https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases) "
                    "or use backend='playwright'."
                ) from e

    if backend in ("auto", "playwright"):
        try:
            return _to_pdf_playwright(html_str, output_path)
        except Exception as e:  # noqa: BLE001
            errors.append(f"playwright: {e}")

    raise RuntimeError(
        "PDF generation failed. Tried backends and got:\n  - "
        + "\n  - ".join(errors)
        + "\n\nFix one of:\n"
        "  • Install GTK3 runtime (for WeasyPrint), or\n"
        "  • Run `python -m playwright install chromium` (for the Playwright fallback)."
    )


def _to_pdf_playwright(html_str: str, output_path: Path) -> Path:
    from playwright.sync_api import sync_playwright  # type: ignore

    _ensure_playwright_chromium()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html_str, wait_until="load")
            page.pdf(
                path=str(output_path),
                format="A4",
                print_background=True,
                margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            )
        finally:
            browser.close()
    return output_path


_CHROMIUM_READY = False


def _ensure_playwright_chromium() -> None:
    """Download Chromium on first PDF render in fresh environments.

    Idempotent: Playwright skips download if the browser is already present.
    Lets the app work on Streamlit Cloud / fresh containers without a manual
    `playwright install chromium` step.
    """
    global _CHROMIUM_READY
    if _CHROMIUM_READY:
        return
    import subprocess
    import sys

    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=False,
            capture_output=True,
            timeout=300,
        )
    except Exception:  # noqa: BLE001
        pass
    _CHROMIUM_READY = True
