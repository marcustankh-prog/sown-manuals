"""Local manual library — save / list / load / delete generated flowers.

Stores each flower as `library/{slug}.json` (the full Flower model JSON).
On Streamlit Community Cloud the filesystem is ephemeral and these files
will not survive container restarts, so users should still download the
JSON spec for permanent backup. For local development, the library
persists between sessions.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from src.models import Flower

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "library"
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "flower").lower()).strip("-")
    return s or "flower"


def _path_for(slug: str) -> Path:
    return LIBRARY_DIR / f"{slug}.json"


def save(flower: Flower, save_name: str | None = None) -> Path:
    """Persist a flower to the library. If `save_name` is given, it is used
    as the filename slug AND stored on the flower as its library label so
    the saved-manuals list reflects the user's chosen name. Overwrites."""
    if save_name and save_name.strip():
        flower.library_label = save_name.strip()
        slug = slugify(save_name)
    else:
        slug = slugify(flower.name)
    p = _path_for(slug)
    p.write_text(flower.model_dump_json(indent=2), encoding="utf-8")
    return p


def load(slug: str) -> Flower:
    return Flower.model_validate_json(_path_for(slug).read_text(encoding="utf-8"))


def delete(slug: str) -> None:
    p = _path_for(slug)
    if p.exists():
        p.unlink()


def exists(slug: str) -> bool:
    return _path_for(slug).exists()


def list_entries() -> list[dict]:
    """Return library entries sorted by last-modified descending."""
    out: list[dict] = []
    for p in LIBRARY_DIR.glob("*.json"):
        try:
            data = Flower.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "slug": p.stem,
            "name": data.library_label or data.name,
            "title": data.title,
            "mtime": p.stat().st_mtime,
            "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "components": len(data.components),
        })
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out
