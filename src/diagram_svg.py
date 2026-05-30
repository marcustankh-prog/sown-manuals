"""Generate component-step and assembly SVG diagrams via Claude.

For French-beading instructions, the diagrams are highly schematic: thin
curved wire paths with rows of small circles representing beads, plus a
short handwritten label. SVG gives us guaranteed-accurate bead counts,
labels, and a controllable background colour.
"""
from __future__ import annotations

import os
import re

from .models import AssemblySection, Component, Flower

# Shared visual contract for both diagram types.
_STYLE_BLOCK = """Visual contract:
- Output a single <svg> element with viewBox="0 0 800 800".
- Background: a <rect> filling the viewBox in #ffffff (pure white).
- All line work in stroke #2b2b2b, stroke-width 1.4-1.8, fill="none"
  (or fill #ffffff for closed shapes).
- Beads: small <circle> elements, r=4 to r=6, stroke #2b2b2b stroke-width
  1.2, fill #ffffff. Beads sit ALONG the wire path, evenly spaced.
- Wire: thin curved <path> (use C/Q bezier) in stroke #2b2b2b stroke-width
  1.4, fill="none". Wires loop, twist, and curve like real beading wire.
- One short handwritten label: <text> in fill #8a6a3f, font-family
  "Caveat, Patrick Hand, Comic Sans MS, cursive", font-size 22,
  font-style italic, placed in the OUTER MARGIN with a thin curved arrow
  (stroke #8a6a3f stroke-width 1.2 fill="none") pointing at the technique.
- A small handwritten step number "1.", "2.", "3." in the top-left corner
  is fine if the diagram shows a numbered step.
- Generous white space. Schematic, NOT photographic.
- At least 100px margin on each side.

Return ONLY the SVG. The first character of your reply must be '<'.
"""

COMPONENT_SYSTEM_PROMPT = (
    "You are a technical illustrator producing French-beading instructional "
    "diagrams as raw SVG markup (no markdown, no commentary, no <html>). "
    "Each diagram shows ONE technique step for making a beaded flower part: "
    "the wire being looped, twisted, or shaped, with the EXACT number of "
    "beads as small circles strung along it.\n\n" + _STYLE_BLOCK
)

ASSEMBLY_SYSTEM_PROMPT = (
    "You are a technical illustrator producing French-beading assembly "
    "diagrams as raw SVG markup (no markdown, no commentary, no <html>). "
    "Each diagram shows how the finished beaded parts attach to a central "
    "stem wire: a vertical stem line with floral-tape wraps shown as short "
    "diagonal hatches, and petal/leaf clusters attaching at staggered "
    "heights as small grouped shapes.\n\n" + _STYLE_BLOCK
)


def _component_user_prompt(flower: Flower, comp: Component, variant: int) -> str:
    heading = comp.heading.rstrip(":").strip()
    label_hint = heading.split("(")[0].strip()
    excerpt = (comp.paragraphs[0] if comp.paragraphs else heading)[:500]

    # Try to extract a small bead count for the diagram (cap so the SVG
    # stays readable; real counts in text can be in the hundreds).
    diagram_bead_count = _suggest_bead_count(excerpt)

    variant_hints = {
        1: "Show the FIRST stage of the technique: wire being shaped into the basic loop or row.",
        2: "Show a LATER stage of the same technique: additional rows or wraps being added.",
        3: "Show the FINISHED part: the completed shape with its beads.",
    }
    variant_hint = variant_hints.get(variant, variant_hints[1])

    return (
        f"Flower: {flower.name}\n"
        f"Component: {heading}\n"
        f"Step description (verbatim from the manual):\n{excerpt}\n\n"
        f"{variant_hint}\n\n"
        f"Show approximately {diagram_bead_count} beads as small circles "
        f"along the wire (this is a visual hint of the technique, not the "
        f"full real count). Place a single handwritten label in the margin "
        f"that reads exactly: \"{label_hint}\". Do not invent other text."
    )


def _assembly_user_prompt(
    flower: Flower, section: AssemblySection, variant: int
) -> str:
    heading = section.heading.rstrip(":").strip()
    excerpt = (section.paragraphs[0] if section.paragraphs else heading)[:500]
    component_names = [
        c.heading.split("(")[0].strip().rstrip(":").strip()
        for c in flower.components
        if c.heading
    ]

    variant_hints = {
        1: "Show an EARLY stage: the bare stem wire with the first one or two parts being attached at the top.",
        2: "Show a LATER stage: more parts attached at staggered heights down the stem, with floral tape wraps visible.",
        3: "Show the FINISHED stem: all parts attached, neatly tape-wrapped, ready to display.",
    }
    variant_hint = variant_hints.get(variant, variant_hints[1])

    parts_block = ", ".join(component_names) if component_names else "petals, leaves"

    return (
        f"Flower: {flower.name}\n"
        f"Assembly step description:\n{excerpt}\n\n"
        f"{variant_hint}\n\n"
        f"The parts being attached to the stem are: {parts_block}. "
        f"Draw the stem vertically down the middle of the canvas. Show "
        f"floral-tape wraps as short diagonal hatch marks along the stem. "
        f"Show petal/leaf clusters as small simple grouped shapes (a few "
        f"loops with bead circles), NOT detailed botanical drawings. "
        f"Place a single handwritten label in the margin that reads exactly: "
        f"\"stem assembly\". Do not invent other text."
    )


def _suggest_bead_count(text: str) -> int:
    """Extract a small bead count from the step text, capped for readability."""
    nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", text)]
    if not nums:
        return 12
    # Prefer small-ish counts (looks like a real bead count, not a row index).
    candidates = [n for n in nums if 3 <= n <= 40]
    chosen = candidates[0] if candidates else min(nums[0], 24)
    return max(6, min(chosen, 24))


def _call_claude(system_prompt: str, user_prompt: str, *, model: str | None) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing — cannot generate SVG.")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    if not text.lstrip().startswith("<svg"):
        m = re.search(r"<svg[\s\S]*?</svg>", text)
        if m:
            text = m.group(0)
        else:
            raise RuntimeError("Claude did not return SVG. Reply was: " + text[:300])
    return text


def generate_component_svg(
    flower: Flower, comp: Component, variant: int = 1, *, model: str | None = None
) -> str:
    return _call_claude(
        COMPONENT_SYSTEM_PROMPT,
        _component_user_prompt(flower, comp, variant),
        model=model,
    )


def generate_assembly_svg(
    flower: Flower,
    section: AssemblySection,
    variant: int = 1,
    *,
    model: str | None = None,
) -> str:
    return _call_claude(
        ASSEMBLY_SYSTEM_PROMPT,
        _assembly_user_prompt(flower, section, variant),
        model=model,
    )
