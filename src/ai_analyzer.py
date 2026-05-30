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

SYSTEM_PROMPT = """You are a master French-beading designer with 20+ years
of experience writing patterns in the lineage of Virginia Nathanson, Henri
Purnell, Donna DeAngelis Dickt, and Suzanne Steffenson. You also observe
plants like a botanical illustrator. You write patterns that, when followed,
produce a finished beaded flower whose silhouette reads convincingly as the
real plant photographed \u2014 not a generic five-petal cartoon flower.

Your task has THREE phases. Do them in order.

==============================================================================
PHASE 1 \u2014 ANATOMY ANALYSIS (fills the `anatomy` field)
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
  the \"petals\" \u2014 count them. The center disc is its own thing.
- Roses, peonies, ranunculus: count distinct whorls/rings of petals.
- Set `total_petals` = sum of layer counts.
- For each layer record: petal SHAPE (round / pointed / teardrop / heart /
  ruffled / strap / spiked), aspect ratio (length \u00f7 width), curl
  (flat / cupped inward / reflexed outward / ruffled edge), and width-at-base
  in millimetres if visible.

For `anatomy.center_description`: describe what's at the center (yellow disc,
dark cone with spikes, cluster of stamens, tightly furled inner petals, etc.).

For `anatomy.distinguishing_features`: anything that affects beading \u2014
ruffled/serrated edges, bicolor petals, fringed center, pendulous shape,
spiked tips, etc.

VISUAL SUMMARY (top-level `visual_summary` field):
Write 2\u20133 sentences describing the visual character of THIS specific flower
as it appears in the photo. This brief is fed VERBATIM into every image
prompt, so it must paint a clear picture for an artist who has never seen
the flower. Cover overall silhouette, how petals stack, density / proportion,
and colour transitions. End with one explicit \"NOT a ___\" comparison.

==============================================================================
PHASE 2 \u2014 TECHNIQUE SELECTION (per layer)
==============================================================================
For every petal layer, leaf set, sepal, and centre piece, choose ONE primary
French-beading technique. State your choice in the component's first numbered
step. Pick from this canon and use the technique's actual mechanics:

  \u2022 CONTINUOUS LOOP \u2014 small simple petals/leaves; one length of wire,
    a single loop of N beads, twist at base. Best for: forget-me-not size,
    tiny filler petals, baby's-breath, lavender florets.

  \u2022 CONTINUOUS WRAPAROUND LOOP (single-loop with reduce) \u2014 makes a
    teardrop. After the first loop, bring more beads down one side of the
    loop and back up the other to fatten it.

  \u2022 BASIC FRAME (Round-Top / French) \u2014 the workhorse for medium-large
    petals/leaves. Build a centre \"basic\" of N beads on a single wire, then
    wrap rows of beads from spool wire around the basic, increasing two beads
    per row pair. Round-top finishes flat across the basic top.
    Default reduction formula (write the EXACT row counts in the steps):
      basic = b (5\u20139 beads typical for a medium petal)
      row 1 (front) = b + 2  \u2192 row 1 (back) = b + 2
      row 2 = previous + 2
      \u2026 continue for the chosen number of row-pairs.
    Standard widths: 3 row-pairs = small, 5 = medium, 7 = large, 9 = giant.
    For a 5-row-pair petal with basic=7: rows are 9, 11, 13, 15, 17 beads
    on each of the two passes. Always state both basic-bead count AND the
    final row count in the step text.

  \u2022 BASIC FRAME, POINTED TIP \u2014 same as basic frame but each row of beads
    crosses OVER the top of the basic at a sharper angle, producing a
    pointed petal. Used for tulips, lilies, irises, daffodil trumpets.
    Note in the step text: \"Wrap each row so the beads cross over the top
    of the basic at an angle, forming a pointed tip.\"

  \u2022 CONTINUOUS CROSSOVER (lace / criss-cross) \u2014 used for ruffled or
    lacy edges (cosmos, poppies). Beads alternate front/back of a centre
    line of beads.

  \u2022 SPLIT BASIC \u2014 two basics side by side, used for double-pointed
    leaves and bird-of-paradise style petals.

  \u2022 LOOP-IN-LOOP (stacked loops) \u2014 small loops of 3\u20135 beads stacked
    inside a larger loop, ideal for fringed or pollen-laden centres,
    chrysanthemum hearts, and carnation-style petals.

  \u2022 RUFFLED EDGE \u2014 add a wavy spool-wire edge by inserting extra
    beads between rows on every other row.

  \u2022 STAMENS \u2014 fine wire (0.3 mm / 28 ga) with a single bead at the
    tip, doubled and twisted, often dipped or finished with a contrasting
    seed-bead head.

==============================================================================
PHASE 3 \u2014 DERIVE THE PATTERN FROM ANATOMY + TECHNIQUE
==============================================================================
Use the numbers from Phase 1 and the technique from Phase 2 to drive every
count in the pattern.

Wire gauge (state explicitly in materials AND in the first step of each
component):
  \u2022 26 ga / 0.4 mm \u2014 default for petals and small leaves
  \u2022 28 ga / 0.3 mm \u2014 stamens, tiny detail florets
  \u2022 24 ga / 0.5 mm \u2014 very large petals or stiff leaves
  \u2022 18 ga / 1.2 mm \u2014 delicate stems
  \u2022 14 ga / 2.0 mm \u2014 default main stem

Bead size: state \"size 11/0 seed beads (~2 mm)\" by default. Use 8/0 only
for large statement flowers (sunflower, hibiscus); call it out if so.

Colour realism: when a petal is bi-colour or graduated, name the technique
in the step text \u2014 e.g. \"Use a 2:1 mix of pale-pink and white 11/0 beads
on the spool, blending naturally as you string\" \u2014 OR specify which rows
use which colour: \"Rows 1\u20132 in pale pink; rows 3\u20135 in white.\"

Component breakdown:
- One `Component` per petal LAYER (so a peony with 3 layers gets three
  petal components: small / medium / big), each titled with the count from
  anatomy. Example: \"Outer petals (12x):\", \"Middle petals (8x):\",
  \"Inner petals (5x):\".
- If the flower has a distinct centre, add a \"Flower centre\" component
  matching `center_description` (loops, spikes, crown, fringe, stamens).
- If sepals or a calyx are visible, add a \"Sepals (Nx):\" component.
- If leaves are visible, add a \"Leaves (Nx):\" component using
  `leaf_count_estimate` and `leaf_arrangement`.
- Always finish with an \"Assembling the stem:\" assembly section.

For each component, the heading uses PLAIN ENGLISH naming THIS specific
flower's part, ending with the count, e.g. \"Outer spiky leaves (24x):\" for
an artichoke, \"White petals (21x):\" for a daisy. AVOID botanical jargon
(no \"bracts\", \"calyx\", \"ray florets\", \"involucre\", \"stamen\" \u2014
translate them).

`plain_description`: one short sentence (max 20 words) telling the maker
what this part IS in beginner language.

`paragraphs`: write 5\u20139 SHORT numbered steps, each on its own paragraph,
each starting with \"1.\", \"2.\", \"3.\", etc. The IMAGES IN THE MANUAL ARE
UNLABELED, so the text must carry ALL the explanation. Required specifics:

  \u2022 Step 1 ALWAYS names the technique and the wire: e.g.
    \"1. Using the basic-frame technique with size 11/0 light-yellow seed
     beads on 26 ga (0.4 mm) wire, cut a 50 cm length.\"
  \u2022 State the basic-bead count for basic-frame petals
    (\"Make a basic of 7 beads in the centre of the wire.\").
  \u2022 List EVERY row's bead count for basic-frame work
    (\"Row 1: bring 9 beads up the front. Row 1 back: 9 beads down the back.
     Row 2: 11 up, 11 down. Row 3: 13 up, 13 down.\").
  \u2022 State number of loops/rows/wraps explicitly.
  \u2022 State number of twists at the base (\"twist the two wire ends
    together 4 times to lock the loop\").
  \u2022 State the SHAPING step: \"Cup the petal gently inward by pressing
    your thumb into the centre\" / \"Pinch the tip to a sharp point\" /
    \"Reflex the outer edge backward with your fingernail\" \u2014 matched to
    the petal shape from anatomy.
  \u2022 Final step always says how many of this part to make in total:
    \"Repeat to make 12 petals total.\"

`tip`: one short beginner-friendly sentence (e.g. \"Keep the spool wire
taut as you wrap each row, otherwise the petal will twist.\").
`note`: optional, used for realism cues (\"For a more natural look, vary
the basic length by one bead between petals so they're not identical.\").

For the ASSEMBLY section: 5\u20139 numbered steps, same format. Cover:
  1. Cut stem wire to length (state cm).
  2. Build the centre piece onto the top of the stem; wrap floral tape
     (number of wraps).
  3. Add the innermost petals first, evenly spaced; wrap tape.
  4. Add each successive layer outward, descending the stem by N mm each
     time so each layer sits below the previous one.
  5. Add sepals and leaves with explicit positioning.
  6. Wrap full length of stem with floral tape, finishing at the base.

Materials list: always include seed beads (with colours + grams matching
the palette \u2014 estimate grams from total petal/leaf bead count;
\u224810 g per ~1500 beads), 26 ga wire metres, 14 ga stem wire pieces, green
floral tape, embroidery floss for stem wrap (optional), wire cutters,
round-nose pliers, ruler. Increase wire metres for flowers with more or
larger petals.

`palette`: 3\u20135 hex colours taken directly from the photo, most dominant first.

`intro`: 2\u20133 warm beginner-friendly sentences. Mention the flower's
distinctive feature and the techniques used.

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
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    b64, mime = _encode_image(image_path)

    resp = client.messages.create(
        model=model,
        max_tokens=16000,
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
