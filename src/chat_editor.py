"""Conversational editor for a Flower manual.

The user chats with Claude while looking at the live preview. Claude can:
  * answer questions about the manual,
  * ask clarifying questions,
  * call the `update_flower` tool to apply text/structural edits,
  * call the `regenerate_images` tool to rebuild illustrations whose look
    no longer matches the manual.

Returns (assistant_text, possibly_updated_flower, updated_history,
regen_requests).
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from .models import Flower

SYSTEM_PROMPT = """You are an expert French-beading designer helping the user
refine a single flower pattern manual. The user is looking at a live preview
of the manual while chatting with you.

You have TWO tools:

1. `update_flower` — use whenever the user asks for ANY change to the manual
   text, counts, components, materials, palette, intro, anatomy, or
   visual_summary. When you call it:
   - Pass the COMPLETE updated Flower JSON (every top-level field). Preserve
     all fields you are not changing exactly as they were, INCLUDING image
     paths in `hero_image`, `anatomy_diagram`, `components[*].images[*].path`,
     `assembly.images[*].path`, and `inspo_images[*].path`.
   - Use plain English in component headings (e.g. "Outer spiky leaves
     (24x):" not "Bracts (24x):"). Avoid botanical jargon.
   - Keep `plain_description` concise — one beginner-friendly sentence per
     component.
   - When the user describes how the flower SHOULD look (silhouette, petal
     stacking, density, colour), update `visual_summary` to reflect that —
     it grounds every illustration prompt.

2. `regenerate_images` — use when an illustration no longer matches the
   manual or the user explicitly asks to regenerate. You can target one
   scope at a time:
     - "anatomy"     — the unlabeled hand-drawn whole-flower diagram
     - "components"  — the per-component technique sketches
     - "assembly"    — the stem-assembly sketches
     - "inspo"       — the photo-style finished beaded replicas
     - "all"         — regenerate every illustration in the manual
   Optional `brief_override`: a 2–3 sentence visual brief used FOR THIS RUN
   ONLY (without mutating the flower's visual_summary). Use it only if the
   user wants a one-off variation. If you also want to update the saved
   visual_summary permanently, call `update_flower` first.

   Typical flow when an illustration is wrong:
   a. Ask one quick clarifying question if the user's intent is ambiguous.
   b. Call `update_flower` to update `visual_summary` (and any other text
      that was wrong).
   c. Call `regenerate_images` with the appropriate scope to rebuild.
   You may call both tools in the SAME turn.

If the user just asks a question, answer in chat without calling any tool.
If something is ambiguous, ask a brief clarifying question instead of
guessing.

Respond conversationally and briefly (2-4 sentences) alongside any tool
calls so the user understands what you did.
"""


def _flower_tool_schema() -> dict:
    return {
        "name": "update_flower",
        "description": (
            "Replace the current flower manual with an updated version. "
            "Pass the COMPLETE Flower JSON, preserving every field you are "
            "not changing (especially image paths)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "flower": Flower.model_json_schema(),
                "summary": {
                    "type": "string",
                    "description": (
                        "One short sentence summarizing what you changed."
                    ),
                },
            },
            "required": ["flower", "summary"],
        },
    }


def _regen_tool_schema() -> dict:
    return {
        "name": "regenerate_images",
        "description": (
            "Regenerate one scope (or all scopes) of illustrations in the "
            "manual using the current Flower state. Existing images in the "
            "selected scope are replaced."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["anatomy", "components", "assembly", "inspo", "all"],
                    "description": "Which illustrations to regenerate.",
                },
                "brief_override": {
                    "type": "string",
                    "description": (
                        "Optional 2-3 sentence visual brief used for this "
                        "run only. Leave empty to use the flower's saved "
                        "visual_summary. If you want the brief saved "
                        "permanently, update visual_summary via "
                        "update_flower instead."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "One short sentence explaining why you're "
                        "regenerating, shown to the user."
                    ),
                },
            },
            "required": ["scope", "reason"],
        },
    }


def chat(
    flower: Flower,
    history: List[dict],
    user_message: str,
    *,
    model: Optional[str] = None,
) -> Tuple[str, Optional[Flower], List[dict], List[dict]]:
    """Send one chat turn to Claude.

    Returns:
        (assistant_reply_text, updated_flower_or_none, new_history,
         regen_requests)

        regen_requests is a list of dicts like
        [{"scope": "inspo", "brief_override": "..."}] for the caller to
        execute after applying any flower update.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing — set it in .env to chat.")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    flower_json = flower.model_dump_json(indent=2)
    system = (
        SYSTEM_PROMPT
        + "\n\n=== CURRENT FLOWER MANUAL (JSON) ===\n"
        + flower_json
    )

    api_messages = list(history) + [{"role": "user", "content": user_message}]

    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        tools=[_flower_tool_schema(), _regen_tool_schema()],
        messages=api_messages,
    )

    reply_text_parts: list[str] = []
    updated: Optional[Flower] = None
    summary: Optional[str] = None
    regen_requests: list[dict] = []

    for block in resp.content:
        if block.type == "text":
            reply_text_parts.append(block.text)
        elif block.type == "tool_use" and block.name == "update_flower":
            try:
                payload = block.input.get("flower", {})
                updated = Flower.model_validate(payload)
                summary = block.input.get("summary")
            except Exception as e:  # noqa: BLE001
                reply_text_parts.append(
                    f"\n\n_(Tried to apply an update but the JSON was invalid: {e})_"
                )
        elif block.type == "tool_use" and block.name == "regenerate_images":
            scope = block.input.get("scope")
            if scope in ("anatomy", "components", "assembly", "inspo", "all"):
                regen_requests.append(
                    {
                        "scope": scope,
                        "brief_override": (block.input.get("brief_override") or "").strip()
                        or None,
                        "reason": block.input.get("reason") or "",
                    }
                )

    reply_text = "\n".join(p for p in reply_text_parts if p).strip()
    if not reply_text:
        reply_text = summary or "Done."
    elif summary and summary not in reply_text:
        reply_text += f"\n\n_Change applied: {summary}_"
    if regen_requests:
        scopes = ", ".join(r["scope"] for r in regen_requests)
        reply_text += f"\n\n_Regenerating illustrations: {scopes}…_"

    new_history = api_messages + [{"role": "assistant", "content": reply_text}]
    return reply_text, updated, new_history, regen_requests
