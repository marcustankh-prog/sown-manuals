"""Generate a labeled anatomy SVG via Claude.

Unlike gpt-image-1, an SVG lets us guarantee that each label points to the
correct part of the diagram, because Claude controls the exact x/y of every
arrow and label.
"""
from __future__ import annotations

import os
import re
from typing import List

from .models import Flower

SYSTEM_PROMPT = """You are a botanical illustrator who outputs ONLY raw SVG
markup (no markdown, no commentary, no <html>) for a labeled anatomy diagram
of a single flower. The diagram is a printed reference that must clearly show
the maker which part of the flower each component corresponds to.

Hard constraints:
- Output a single <svg> element with viewBox="0 0 800 1100" (portrait).
- Background: a <rect> filling the viewBox in #ffffff (pure white).
- All flower line work in stroke #2b2b2b, stroke-width 1.5–2, fill="none"
  (or fill #ffffff for closed shapes that should look like white paper).
- Labels: <text> in fill #8a6a3f, font-family "Caveat, Patrick Hand, Comic Sans MS, cursive",
  font-size 22, font-style italic. Position labels in the OUTER MARGINS of
  the canvas (left side, right side, top, or bottom) so they never overlap
  the drawing.
- Each label has a thin curved <path> arrow in stroke #8a6a3f, stroke-width 1.2,
  fill="none", marker-end pointing to the EXACT centre of the part being
  labelled. Use a <defs><marker> for a small triangular arrowhead.
- The labels you draw MUST be the EXACT component names provided in the user
  message, one label per component name. Do not invent or rename.
- Keep the drawing schematic, NOT photographic. Concentric layers can be
  ellipses or simple lobed shapes. Use 3\u20136 distinct elements depending on
  what the flower has (outer petal ring, middle ring, inner ring, centre,
  stem, leaves, etc.). Keep it readable and uncluttered.
- Center the drawing in the canvas with at least 140px margin on each side
  so labels have room.

Return ONLY the SVG. The first character of your reply must be '<'.
"""


def _user_prompt(flower: Flower, component_labels: List[str]) -> str:
    anatomy_lines: List[str] = []
    if flower.anatomy:
        a = flower.anatomy
        if a.overall_shape:
            anatomy_lines.append(f"- overall shape: {a.overall_shape}")
        if a.petal_layers:
            for layer in a.petal_layers:
                anatomy_lines.append(
                    f"- {layer.name} layer: {layer.count} petals"
                    + (f", {layer.shape}" if layer.shape else "")
                )
        if a.center_description:
            anatomy_lines.append(f"- centre: {a.center_description}")
        if a.leaf_count_estimate:
            anatomy_lines.append(f"- leaves: ~{a.leaf_count_estimate}")
    anatomy_block = "\n".join(anatomy_lines) or "(no extra anatomy provided)"

    label_block = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(component_labels))

    return (
        f"Flower: {flower.name}\n\n"
        f"Anatomy summary:\n{anatomy_block}\n\n"
        f"Draw and label exactly these {len(component_labels)} parts, "
        f"using these strings VERBATIM as the label text:\n{label_block}\n\n"
        f"Place the labels around the diagram so each arrow clearly points to "
        f"the corresponding shape (outermost ring for an 'outer' label, the "
        f"centre for a 'centre' / 'crown' label, the stem for a 'stem' label, "
        f"leaves for a 'leaves' label, etc.)."
    )


def generate_svg(flower: Flower, *, model: str | None = None) -> str:
    """Return raw SVG markup for the labeled anatomy diagram."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing \u2014 cannot generate SVG.")

    from anthropic import Anthropic

    labels = []
    for comp in flower.components:
        name = comp.heading.split("(")[0].strip().rstrip(":").strip()
        if name:
            labels.append(name)
    if not labels:
        labels = ["flower", "stem", "leaves"]

    client = Anthropic(api_key=api_key)
    model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_prompt(flower, labels)}],
    )

    text = "".join(b.text for b in resp.content if b.type == "text").strip()

    # Be forgiving if Claude wraps in a code fence.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    if not text.lstrip().startswith("<svg"):
        # Try to find the first <svg ...>...</svg> block.
        m = re.search(r"<svg[\s\S]*?</svg>", text)
        if m:
            text = m.group(0)
        else:
            raise RuntimeError("Claude did not return SVG. Reply was: " + text[:300])
    return text
