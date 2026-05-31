"""Generate beading-process illustrations via OpenAI's image API.

Two distinct styles:
  - **Instruction images** (anatomy, components, assembly) â€” minimalist
    hand-drawn black-ink line illustrations on a PURE WHITE background.
    No text, no labels, no numbers, no handwriting anywhere. Beads are
    drawn as small open circles strung along curving wire lines. The text
    of the manual carries all explanation; the image is purely visual.
  - **Inspo gallery images** â€” warm natural-light photographs of the
    finished beaded flower in vases, hands, or on wooden surfaces.
"""
from __future__ import annotations

import base64
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable, Optional

from .models import AssemblySection, Component, ComponentImage, Flower

# Module-level buffer of human-readable errors from the most recent
# generate_for_flower() call. The Streamlit UI reads this so per-image API
# failures (rate limits, org-verification, etc.) are visible instead of only
# going to stderr.
LAST_ERRORS: list[str] = []


def _log_image_error(label: str, exc: BaseException) -> None:
    """Print image-API errors to stderr (visible in the Streamlit terminal)."""
    msg = f"{label}: {type(exc).__name__}: {exc}"
    LAST_ERRORS.append(msg)
    print(
        f"[image_generator] {label} FAILED: {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    traceback.print_exception(type(exc), exc, exc.__traceback__)

# ---------------------------------------------------------------------------
# Style snippets
# ---------------------------------------------------------------------------

LINE_ART_STYLE = (
    "Style: HAND-DRAWN black-ink illustration, sketched by a human with a "
    "fine ink pen on PURE WHITE (#ffffff) paper. The lines must look "
    "hand-drawn: slightly loose and imperfect, with subtle pen-pressure "
    "variation, gentle wobbles, and small organic irregularities \u2014 NOT "
    "perfectly straight, NOT vector art, NOT CAD, NOT digital-clean. "
    "Like a page from a vintage 1970s French-beading pattern booklet "
    "(Henri Purnell / Virginia Nathanson aesthetic). Seed beads are drawn "
    "as small hand-sketched open circles strung in chains along thin "
    "curving wire lines. Black ink only \u2014 no color, no shading, no "
    "gradients, no textured paper, no cross-hatching. Generous white space. "
    "ABSOLUTELY NO text, NO letters, NO numbers, NO handwriting, NO labels, "
    "NO captions, NO watermarks, NO arrows pointing to labels. The image "
    "is purely visual. No photographic detail, no environment, no human "
    "figures."
)

PHOTO_STYLE = (
    "Style: warm natural-daylight still-life photograph, soft shadows, "
    "muted cream and beige tones, magazine-quality craft styling. Shallow "
    "depth of field. No text, no captions, no watermarks."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client():
    from openai import OpenAI
    return OpenAI()


def _generate_one(prompt: str, *, model: str, size: str = "1024x1024") -> bytes:
    resp = _client().images.generate(model=model, prompt=prompt, n=1, size=size)
    b64 = resp.data[0].b64_json
    if b64:
        return base64.b64decode(b64)
    url = resp.data[0].url
    import urllib.request
    with urllib.request.urlopen(url) as r:  # noqa: S310
        return r.read()


def _save(image_bytes: bytes, out_dir: Path, basename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basename}.png"
    path.write_bytes(image_bytes)
    return path


def _generate_from_reference(
    reference_paths: list[Path],
    prompt: str,
    *,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
) -> bytes:
    """Generate an image conditioned on one or more reference photos.

    Uses OpenAI's image-edit endpoint so gpt-image-1 can use the original
    upload as a visual anchor (subject silhouette, palette, framing).
    The endpoint only accepts PNG inputs (≤4MB, square preferred), so we
    re-encode each reference into a 1024×1024 RGBA PNG first.
    """
    from PIL import Image, ImageOps
    from io import BytesIO
    import tempfile

    prepared: list[Path] = []
    tmp_files: list[tempfile._TemporaryFileWrapper] = []
    try:
        for p in reference_paths:
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGBA")
            # Pad to square on transparent background, then resize.
            side = max(img.width, img.height)
            canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            canvas.paste(img, ((side - img.width) // 2,
                               (side - img.height) // 2), img)
            canvas = canvas.resize((1024, 1024), Image.LANCZOS)
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            )
            canvas.save(tmp.name, format="PNG", optimize=True)
            tmp.close()
            prepared.append(Path(tmp.name))
            tmp_files.append(tmp)

        files = [open(p, "rb") for p in prepared]
        try:
            resp = _client().images.edit(
                model=model,
                image=files if len(files) > 1 else files[0],
                prompt=prompt,
                n=1,
                size=size,
            )
        finally:
            for f in files:
                try:
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
    finally:
        for p in prepared:
            try:
                p.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    b64 = resp.data[0].b64_json
    if b64:
        return base64.b64decode(b64)
    url = resp.data[0].url
    import urllib.request
    with urllib.request.urlopen(url) as r:  # noqa: S310
        return r.read()


def _hero_prompt(flower: Flower, brief: str = "") -> str:
    palette = ", ".join(flower.palette[:4]) if flower.palette else "natural muted"
    brief_block = (
        f"The beaded replica's silhouette and petal arrangement must match "
        f"this visual reference: {brief} "
        if brief
        else ""
    )
    return (
        f"Editorial cover photograph of a FINISHED BEADED REPLICA of a "
        f"{flower.name.lower()} flower — hand-crafted from many tiny glass "
        f"seed beads on thin wire (NOT a real living plant, NOT a silk "
        f"flower). {brief_block}"
        f"The flower is the clear hero of the frame on a soft cream paper "
        f"background with gentle natural light, plenty of negative space "
        f"around it for cover typography. You can clearly see the individual "
        f"seed-bead texture. Color palette: {palette}. {PHOTO_STYLE}"
    )


def generate_from_reference(
    flower: Flower,
    target: str,
    reference_paths: list[Path],
    *,
    out_dir: Path,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    brief_override: Optional[str] = None,
) -> Path:
    """Generate an image for `target` (hero/anatomy/inspo) from references.

    Returns the saved PNG path. Caller is responsible for attaching it to
    the flower model.
    """
    brief = _visual_brief(flower, override=brief_override)
    if target == "hero":
        prompt = _hero_prompt(flower, brief=brief)
        basename = "hero_ai"
    elif target == "anatomy":
        prompt = _anatomy_diagram_prompt(flower, brief=brief)
        basename = "anatomy_ai"
    elif target == "inspo":
        idx = len(flower.inspo_images)
        prompt = _inspo_prompt(flower, idx=idx, brief=brief)
        basename = f"inspo_ai_{idx + 1}"
    else:
        raise ValueError(f"Unsupported target for reference generation: {target}")

    img_bytes = _generate_from_reference(
        reference_paths, prompt, model=model, size=size
    )
    return _save(img_bytes, out_dir, basename)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _visual_brief(flower: Flower, override: Optional[str] = None) -> str:
    """Return a short paragraph describing how THIS flower looks.

    Used to ground every illustration prompt so gpt-image-1 doesn't fall back
    on a generic flower shape. `override` (from chat) wins when present.
    """
    parts: list[str] = []
    if override and override.strip():
        parts.append(override.strip())
    elif flower.visual_summary:
        parts.append(flower.visual_summary.strip())
    if flower.anatomy and flower.anatomy.overall_shape and not parts:
        parts.append(f"Overall shape: {flower.anatomy.overall_shape}.")
    if flower.palette:
        parts.append("Colour palette: " + ", ".join(flower.palette[:5]) + ".")
    return " ".join(parts).strip()


def _component_prompt(
    flower: Flower, comp: Component, variant: int, brief: str = ""
) -> str:
    excerpt = (comp.paragraphs[0] if comp.paragraphs else comp.heading)[:380]
    heading = comp.heading.rstrip(":")
    brief_block = f"Visual reference for this flower: {brief} " if brief else ""
    return (
        f"A HAND-DRAWN sketch (black ink pen on white paper, vintage craft "
        f"booklet style) showing how to make the '{heading}' part of a "
        f"French-beaded {flower.name.lower()}. {brief_block}"
        f"Depict the technique itself \u2014 wire being looped, twisted, or "
        f"shaped, with rows or loops of small hand-sketched open-circle seed "
        f"beads along the wire. The shape of the petal/leaf being formed "
        f"should match the visual reference above. "
        f"Variation {variant}: show the technique from a different angle or step. "
        f"Step description: {excerpt} "
        f"{LINE_ART_STYLE}"
    )


def _assembly_prompt(
    flower: Flower, section: AssemblySection, variant: int, brief: str = ""
) -> str:
    excerpt = (section.paragraphs[0] if section.paragraphs else section.heading)[:380]
    brief_block = f"Visual reference for this flower: {brief} " if brief else ""
    return (
        f"A HAND-DRAWN sketch (black ink pen on white paper, vintage craft "
        f"booklet style) of assembling a French-beaded {flower.name.lower()}. "
        f"{brief_block}"
        f"Show a stem wire being wrapped with floral tape and petal/leaf sets "
        f"attached so that the FINISHED silhouette matches the visual "
        f"reference above (do not default to a daisy or sunflower shape). "
        f"Use thin curving hand-drawn wire lines and small hand-sketched "
        f"open-circle beads. Variation {variant}: focus on a different stage "
        f"of assembly. Step description: {excerpt} "
        f"{LINE_ART_STYLE}"
    )


def _anatomy_diagram_prompt(flower: Flower, brief: str = "") -> str:
    """A clean unlabeled hand-drawn botanical sketch of the whole flower."""
    brief_block = f"Visual reference: {brief} " if brief else ""
    return (
        f"A HAND-DRAWN botanical sketch (black ink pen on white paper, "
        f"vintage botanical-illustration style) of a single complete "
        f"{flower.name.lower()} flower viewed from the front, with its stem "
        f"and a few leaves. {brief_block}"
        f"The drawing must look hand-drawn with loose organic ink lines, NOT "
        f"vector or digital art. The silhouette and petal arrangement must "
        f"match the visual reference above. The whole plant is centred in "
        f"the frame with generous white space around it. {LINE_ART_STYLE}"
    )


def _inspo_prompt(flower: Flower, idx: int, brief: str = "") -> str:
    palette = ", ".join(flower.palette[:4]) if flower.palette else "natural muted"
    settings = [
        "arranged in a small handmade ceramic vase on a wooden table beside a sunlit window",
        "lying flat on natural linen fabric next to a pair of small craft scissors and a spool of fine beading wire",
        "held in a person's hand against a soft cream wall with warm afternoon light",
        "in a slim glass bottle on a wooden shelf, casting a gentle shadow on the wall",
        "grouped with two or three other beaded flowers in a small bouquet on a beige tablecloth",
        "close-up overhead shot on cream paper, showing fine bead detail",
    ]
    setting = settings[idx % len(settings)]
    brief_block = (
        f"The beaded replica's silhouette and petal arrangement must match "
        f"this visual reference: {brief} "
        if brief
        else ""
    )
    return (
        f"Photograph of a FINISHED BEADED REPLICA of a {flower.name.lower()} "
        f"flower \u2014 the entire flower is hand-crafted from many tiny "
        f"glass seed beads strung on thin wire (NOT a real living plant, "
        f"NOT a silk flower). {brief_block}"
        f"The beaded flower is {setting}. It is the clear focal point, fully "
        f"visible in frame, and you can clearly see the individual seed-bead "
        f"texture covering every petal and leaf. Color palette: {palette}. "
        f"{PHOTO_STYLE}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SCOPES_ALL = ("anatomy", "components", "assembly", "inspo")


def generate_for_flower(
    flower: Flower,
    *,
    out_dir: Path,
    images_per_component: int = 2,
    assembly_images: int = 2,
    inspo_images: int = 4,
    model: str | None = None,
    progress=None,
    scopes: Optional[Iterable[str]] = None,
    brief_override: Optional[str] = None,
) -> Flower:
    """Populate flower images for the given scopes.

    Args:
        scopes: subset of ("anatomy", "components", "assembly", "inspo").
            Defaults to all four. When a scope is selected, its existing
            images are replaced with freshly generated ones.
        brief_override: optional visual-brief string that overrides
            `flower.visual_summary` for this run only (used by the chat to
            pass a refined description without mutating the flower).
        progress: callable (label: str, frac: float) for UI updates.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing \u2014 add it to .env.")
    model = model or os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

    LAST_ERRORS.clear()

    active = set(scopes) if scopes else set(SCOPES_ALL)
    invalid = active - set(SCOPES_ALL)
    if invalid:
        raise ValueError(f"Unknown scope(s): {sorted(invalid)}")

    brief = _visual_brief(flower, override=brief_override)

    total = (
        (1 if "anatomy" in active else 0)
        + (len(flower.components) * images_per_component if "components" in active else 0)
        + (assembly_images if "assembly" in active and flower.assembly else 0)
        + (inspo_images if "inspo" in active else 0)
    )
    done = 0

    def _tick(label: str):
        nonlocal done
        done += 1
        if progress and total:
            progress(label, done / total)

    if "anatomy" in active:
        try:
            prompt = _anatomy_diagram_prompt(flower, brief=brief)
            img_bytes = _generate_one(prompt, model=model)
            path = _save(img_bytes, out_dir, "anatomy_diagram")
            flower.anatomy_diagram = ComponentImage(
                path=path.as_uri(),
                caption="The parts you'll be making",
            )
        except Exception as exc:  # noqa: BLE001
            _log_image_error("Anatomy diagram", exc)
        _tick("Anatomy diagram")

    if "components" in active:
        for i, comp in enumerate(flower.components):
            comp.images = []
            for j in range(images_per_component):
                try:
                    prompt = _component_prompt(flower, comp, variant=j + 1, brief=brief)
                    img_bytes = _generate_one(prompt, model=model)
                    path = _save(img_bytes, out_dir, f"comp{i+1:02d}_{j+1}")
                    comp.images.append(ComponentImage(path=path.as_uri()))
                except Exception as exc:  # noqa: BLE001
                    _log_image_error(f"Component {i+1} image {j+1}", exc)
                _tick(f"Component {i+1} image {j+1}")

    if "assembly" in active and flower.assembly is not None:
        flower.assembly.images = []
        for j in range(assembly_images):
            try:
                prompt = _assembly_prompt(
                    flower, flower.assembly, variant=j + 1, brief=brief
                )
                img_bytes = _generate_one(prompt, model=model)
                path = _save(img_bytes, out_dir, f"assembly_{j+1}")
                flower.assembly.images.append(ComponentImage(path=path.as_uri()))
            except Exception as exc:  # noqa: BLE001
                _log_image_error(f"Assembly image {j+1}", exc)
            _tick(f"Assembly image {j+1}")

    if "inspo" in active:
        flower.inspo_images = []
        for j in range(inspo_images):
            try:
                prompt = _inspo_prompt(flower, j, brief=brief)
                img_bytes = _generate_one(prompt, model=model)
                path = _save(img_bytes, out_dir, f"inspo_{j+1}")
                flower.inspo_images.append(ComponentImage(path=path.as_uri()))
            except Exception as exc:  # noqa: BLE001
                _log_image_error(f"Inspo image {j+1}", exc)
            _tick(f"Inspo image {j+1}")

    return flower
