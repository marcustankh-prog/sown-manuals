"""AI photo analyzer.

Given a flower photo, return a draft `Flower` (dominant colors + suggested
beading components and materials). The result is a *starting point* — the
user must review and adjust counts/wire gauges before publishing.

Two providers are supported via env vars:
  - openai (default): uses gpt-4o / gpt-4o-mini with vision
  - anthropic: uses claude-3-5-sonnet with vision

Set AI_PROVIDER in .env to pick.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

from .models import Flower

SYSTEM_PROMPT = """You are an expert French-beading designer AND a careful botanical
observer. You write step-by-step beaded-flower patterns in the style of Henri
Purnell's "Spring Bouquet".

Your task has TWO phases. Do them in order.

==============================================================================
PHASE 1 — ANATOMY ANALYSIS (fills the `anatomy` field)
==============================================================================
Look at the photo carefully and document the flower's structure as a botanist
would, then translate it into beader-relevant numbers. Be specific and honest:
if a value isn't visible in the photo, estimate conservatively but say so in
`distinguishing_features`.

For `anatomy.petal_layers`:
- List layers from OUTERMOST to INNERMOST.
- Count visible petals in each layer. If the flower is symmetric and only
  half is visible, double the visible count and note this.
- Composite flowers (daisy, coneflower, sunflower): the outer ray florets ARE
  the "petals" — count them. The center disc is its own thing.
- Roses, peonies, ranunculus: count distinct whorls/rings of petals.
- Set `total_petals` = sum of layer counts.

For `anatomy.center_description`: describe what's at the center (yellow disc,
dark cone with spikes, cluster of stamens, tightly furled inner petals, etc.).

For `anatomy.distinguishing_features`: anything that affects beading —
ruffled/serrated edges, bicolor petals, fringed center, pendulous shape,
spiked tips, etc.

VISUAL SUMMARY (top-level `visual_summary` field):
Write 2–3 sentences describing the visual character of THIS specific flower
as it appears in the photo. This brief is fed VERBATIM into every image
prompt, so it must paint a clear picture for an artist who has never seen
the flower. Cover:
  - Overall silhouette / shape (torpedo, globe, daisy fan, bell, spike, cup,
    cone, dense head, etc.) — be specific and use a strong noun.
  - How petals stack or overlap (like roof shingles, in concentric rings,
    fanning outward, cupped inward, in a tight spiral, in pendulous strings).
  - Density and proportions (chunky and compact vs. airy and open; taller
    than wide vs. wider than tall).
  - Where colour transitions happen on the flower (base to tip, centre to
    edge, bicolour ring, etc.).
  - End the brief with one explicit "NOT a ___" comparison to ward off
    obvious wrong shapes (e.g. "NOT a fanned-out daisy", "NOT a thin spike").
Example: "A dense torpedo-shaped flower head with overlapping waxy bracts
stacked like roof shingles, wider at the base than the tip. Deep magenta at
the base fading to pale yellow-green at the growing tip. Compact and chunky,
NOT a fanned-out daisy or thin sunflower-style spike."

==============================================================================
PHASE 2 — DERIVE THE PATTERN FROM THE ANATOMY
==============================================================================
Use the numbers from Phase 1 to drive every count in the pattern. Concretely:

- One `Component` per petal LAYER (so a peony with 3 layers gets three
  petal components: small/medium/big), each titled with the count from anatomy.
  Example: "Outer Petals (12x):", "Middle Petals (8x):", "Inner Petals (5x):".
- If the flower has a distinct center, add a "Flower Center" component that
  matches `center_description` (loops, spikes, crown, fringe, etc.).
- If the flower has visible sepals or a calyx, add a "Sepals (Nx):" component.
- If leaves are visible on the stem, add a "Leaves (Nx):" component using
  `leaf_count_estimate` and `leaf_arrangement`.
- Always finish with an "Assembling the Stem:" assembly section that walks
  through joining the layers in the same outer-to-inner order, then leaves.
  The assembly `paragraphs` field follows the SAME numbered-step format as
  the components (4–8 short numbered steps starting with action verbs,
  explicit measurements like "Cut a 35 cm length of 2.0 mm stem wire.",
  "Hold the centre piece against the top of the stem and wrap floral tape
  4 times around the join.", "Add the inner petals 1 cm below the centre
  and wrap 6 times.", etc.).

For each component:
- Heading uses PLAIN ENGLISH naming THIS specific flower's part, ending with
  the count, e.g. "Outer spiky leaves (24x):" for an artichoke,
  "White petals (21x):" for a daisy, "Cupped pink petals (8x):" for a peony.
  NEVER use the literal word "Daisy" unless the photo shows a daisy.
  AVOID botanical jargon — do NOT use words like "bracts", "calyx", "ray
  florets", "involucre", "stamen". Translate them: "bracts" -> "scaly outer
  leaves"; "ray florets" -> "long thin petals"; "stamens" -> "fuzzy yellow
  centre threads".
- `plain_description`: one short sentence (max 20 words) telling the maker
  what this part IS, in language a beginner would understand. Example:
  "These are the long pointed scales that overlap around the outside of the
  artichoke head."
- `paragraphs`: write the instructions as 4–7 SHORT numbered steps, each on
  its own paragraph, each starting with "1.", "2.", "3.", etc. Each step is
  ONE clear sentence beginning with an action verb (Cut, Thread, Bend,
  Twist, Loop, Wrap, Slide). The IMAGES IN THE MANUAL ARE UNLABELED, so the
  text must carry ALL the explanation — be explicit about every measurement
  and count. Required specifics in the steps:
    • Cut wire length in cm (e.g. "Cut a 50 cm length of 0.4 mm wire.")
    • Exact number of beads to thread for each loop or row (e.g.
      "Thread 14 beads onto the wire and slide them to the centre.")
    • Number of loops, rows, or wraps (e.g. "Make 3 more loops the same
      size, twisting the wires together at the base each time.")
    • Number of twists (e.g. "Twist the two wire ends together 4 times to
      lock the loop.")
    • Where to bend or shape (e.g. "Bend the loop into a teardrop, pointed
      at the tip.")
    • How many of this part to make in total (e.g. "Repeat to make 21
      petals total.")
- 0.4 mm wire for petals/leaves; 2.0 mm stem wire (1.2 mm for delicate stems).
- 11/0 seed beads (~2.0 mm) by default.
- Include one short `tip` (one sentence, beginner-friendly) and optionally
  one `note` per component.

For materials: always include seed beads (with colors + grams matching the
palette), 0.4 mm wire, 2.0 mm stem wire (or 1.2 mm), green floral tape,
embroidery floss, scissors, measurement tape. Increase quantities for flowers
with more or larger petals.

For `palette`: 3–5 hex colors taken directly from the photo, most dominant
first.

For `intro`: 2–3 warm beginner-friendly sentences. Mention the flower's
distinctive feature.

Do NOT invent image paths; leave images arrays empty. Output strictly valid
JSON conforming to the schema. No prose outside the JSON.
"""


def _flower_json_schema() -> dict:
    """JSON schema derived from the Flower pydantic model."""
    return Flower.model_json_schema()


def _encode_image(image_path: Path) -> tuple[str, str]:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ext = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
    return b64, f"image/{mime}"


def analyze_photo(image_path: Path, *, hint_name: Optional[str] = None) -> Flower:
    """Send the photo to the configured vision model and return a Flower draft."""
    image_path = Path(image_path)
    provider = os.getenv("AI_PROVIDER", "anthropic").lower()
    user_prompt = (
        "Analyze the flower in the attached photo and produce a complete pattern "
        "manual JSON. Match the schema exactly.\n\n"
        f"Schema:\n{json.dumps(_flower_json_schema())}\n"
    )
    if hint_name:
        user_prompt += f"\nThe user calls this flower: {hint_name}.\n"

    if provider == "openai":
        return _analyze_openai(image_path, user_prompt)
    return _analyze_anthropic(image_path, user_prompt)


# ----------------------------------------------------------------------------
# OpenAI
# ----------------------------------------------------------------------------

def _analyze_openai(image_path: Path, user_prompt: str) -> Flower:
    from openai import OpenAI  # lazy import

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    b64, mime = _encode_image(image_path)
    data_url = f"data:{mime};base64,{b64}"

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    return Flower.model_validate_json(raw)


# ----------------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------------

def _analyze_anthropic(image_path: Path, user_prompt: str) -> Flower:
    import anthropic  # lazy import

    client = anthropic.Anthropic()
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    b64, mime = _encode_image(image_path)

    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT + "\nReturn ONLY a JSON object, no surrounding text.",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64},
                    },
                ],
            }
        ],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    # Strip code-fence guardrails if model added any
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    return Flower.model_validate_json(text)
