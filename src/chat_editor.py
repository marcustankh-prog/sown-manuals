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

import os
from typing import List, Optional, Tuple

from .models import Flower

SYSTEM_PROMPT = """You are an expert French-beading designer helping the user
refine a single flower pattern manual. The user is looking at a live preview
of the manual while chatting with you.

You have FOUR tools:

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

3. `attach_step_image` — use when the user attaches a photo or sketch and
   wants it placed inside a component. The user's attached images are
   listed below the manual JSON as numbered references (image #1, image
   #2, …). Pick the right `component_index` (0-based). Pass `step_index`
   (0-based) when the photo illustrates one specific paragraph; omit
   `step_index` when the photo illustrates the component as a whole. You
   may also write a short `caption`. If the target component is unclear,
   ask one quick clarifying question first instead of guessing.

4. `set_hero_image` — use when the user attaches an image and wants it as
   the cover (hero) image of the manual. Pass the matching `image_id`.

5. `add_inspo_image` — use when the user attaches a finished-piece photo
   and wants it added to the back-of-manual inspo gallery. Pass the
   matching `image_id`.

6. `set_anatomy_diagram` — use when the user attaches a labeled sketch or
   reference for the flower-anatomy diagram. Pass the matching `image_id`.

7. `add_assembly_image` — use when the user attaches a photo for the final
   stem-assembly section. Pass the matching `image_id`; optionally pass
   `step_index` (0-based) within the assembly paragraphs.

For ANY user-uploaded image (hero, anatomy, components, assembly, inspo)
you MUST use one of tools 3–7 to place it. Never edit image paths
(`hero_image`, `anatomy_diagram.path`, `components[*].images`,
`assembly.images`, `inspo_images`) via `update_flower` — only the
dedicated tools above can resolve uploads to real file paths.

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


def _attach_image_tool_schema() -> dict:
    return {
        "name": "attach_step_image",
        "description": (
            "Attach an image the user has uploaded in this turn to a "
            "specific step of a specific component in the manual."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": (
                        "1-based id of the user-uploaded image to attach "
                        "(see the 'ATTACHED IMAGES' list)."
                    ),
                },
                "component_index": {
                    "type": "integer",
                    "description": (
                        "0-based index into flower.components for the "
                        "component this image illustrates."
                    ),
                },
                "step_index": {
                    "type": "integer",
                    "description": (
                        "0-based index of the paragraph inside the "
                        "component this image illustrates. Omit if the "
                        "image illustrates the component as a whole rather "
                        "than one specific step."
                    ),
                },
                "caption": {
                    "type": "string",
                    "description": "Optional one-line caption.",
                },
            },
            "required": ["image_id", "component_index"],
        },
    }


def _hero_tool_schema() -> dict:
    return {
        "name": "set_hero_image",
        "description": (
            "Set one of the user-uploaded images as the cover (hero) "
            "image of the manual."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": (
                        "1-based id of the user-uploaded image to use "
                        "as the cover."
                    ),
                },
            },
            "required": ["image_id"],
        },
    }


def _add_inspo_tool_schema() -> dict:
    return {
        "name": "add_inspo_image",
        "description": (
            "Append one of the user-uploaded images to the back-of-manual "
            "inspo gallery (flower.inspo_images). Use this when the user "
            "wants a finished-piece photo shown as inspiration. Do NOT "
            "edit inspo_images via update_flower — only this tool can "
            "resolve the uploaded image to a real file path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": (
                        "1-based id of the user-uploaded image to add."
                    ),
                },
                "caption": {
                    "type": "string",
                    "description": "Optional one-line caption.",
                },
            },
            "required": ["image_id"],
        },
    }


def _set_anatomy_tool_schema() -> dict:
    return {
        "name": "set_anatomy_diagram",
        "description": (
            "Set one of the user-uploaded images as the anatomy diagram "
            "(flower.anatomy_diagram). Use when the user uploads a labeled "
            "sketch or reference showing the flower's parts. Do NOT edit "
            "anatomy_diagram via update_flower — only this tool resolves "
            "the upload to a real file path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": "1-based id of the user-uploaded image.",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional one-line caption.",
                },
            },
            "required": ["image_id"],
        },
    }


def _add_assembly_tool_schema() -> dict:
    return {
        "name": "add_assembly_image",
        "description": (
            "Append one of the user-uploaded images to the assembly section "
            "(flower.assembly.images). Use for photos illustrating final "
            "stem assembly steps. Do NOT edit assembly.images via "
            "update_flower — only this tool resolves the upload to a real "
            "file path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "integer",
                    "description": "1-based id of the user-uploaded image.",
                },
                "step_index": {
                    "type": "integer",
                    "description": (
                        "Optional 0-based paragraph index within the "
                        "assembly section this image illustrates."
                    ),
                },
                "caption": {
                    "type": "string",
                    "description": "Optional one-line caption.",
                },
            },
            "required": ["image_id"],
        },
    }


def chat(
    flower: Flower,
    history: List[dict],
    user_message: str,
    *,
    attached_images: Optional[List[dict]] = None,
    model: Optional[str] = None,
) -> Tuple[str, Optional[Flower], List[dict], List[dict], List[dict]]:
    """Send one chat turn to Claude.

    `attached_images`: optional list of dicts describing images the user
    uploaded with this turn. Each dict must contain at least `id`
    (1-based int), `b64` (base64-encoded bytes), and `mime`
    (e.g. 'image/png').

    Returns:
        (assistant_reply_text, updated_flower_or_none, new_history,
         regen_requests, image_actions)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing — set it in .env to chat.")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    flower_json = flower.model_dump_json(indent=2)
    lang = (getattr(flower, "language", "en") or "en").lower()
    if lang.startswith("ko"):
        lang_directive = (
            "\n\n=== OUTPUT LANGUAGE ===\n"
            "This manual is written in Korean. Reply to the user in Korean and "
            "keep all manual content (component headings, paragraphs, tips, "
            "materials, assembly steps) in Korean when you call the "
            "`update_flower` tool."
        )
    else:
        lang_directive = ""

    attach_block = ""
    if attached_images:
        lines = [
            f"  - image #{img['id']} ({img.get('mime', 'image/png')})"
            for img in attached_images
        ]
        attach_block = (
            "\n\n=== ATTACHED IMAGES (this turn) ===\n"
            + "\n".join(lines)
            + "\n\nUse `attach_step_image`, `set_hero_image`, `add_inspo_image`, "
            "`set_anatomy_diagram`, or `add_assembly_image` to place "
            "these. The image_id refers to the numbered list above."
        )

    system = (
        SYSTEM_PROMPT
        + lang_directive
        + "\n\n=== CURRENT FLOWER MANUAL (JSON) ===\n"
        + flower_json
        + attach_block
    )

    user_content: list = []
    if attached_images:
        for img in attached_images:
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("mime", "image/png"),
                    "data": img["b64"],
                },
            })
    user_content.append({"type": "text", "text": user_message or "(no text)"})

    api_messages = list(history) + [{"role": "user", "content": user_content}]

    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        tools=[
            _flower_tool_schema(),
            _regen_tool_schema(),
            _attach_image_tool_schema(),
            _hero_tool_schema(),
            _add_inspo_tool_schema(),
            _set_anatomy_tool_schema(),
            _add_assembly_tool_schema(),
        ],
        messages=api_messages,
    )

    reply_text_parts: list[str] = []
    updated: Optional[Flower] = None
    summary: Optional[str] = None
    regen_requests: list[dict] = []
    image_actions: list[dict] = []

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
        elif block.type == "tool_use" and block.name == "attach_step_image":
            try:
                raw_step = block.input.get("step_index")
                image_actions.append({
                    "action": "attach_step",
                    "image_id": int(block.input.get("image_id")),
                    "component_index": int(block.input.get("component_index")),
                    "step_index": int(raw_step) if raw_step is not None else None,
                    "caption": (block.input.get("caption") or "").strip() or None,
                })
            except Exception as e:  # noqa: BLE001
                reply_text_parts.append(
                    f"\n\n_(attach_step_image rejected: {e})_"
                )
        elif block.type == "tool_use" and block.name == "set_hero_image":
            try:
                image_actions.append({
                    "action": "set_hero",
                    "image_id": int(block.input.get("image_id")),
                })
            except Exception as e:  # noqa: BLE001
                reply_text_parts.append(
                    f"\n\n_(set_hero_image rejected: {e})_"
                )
        elif block.type == "tool_use" and block.name == "add_inspo_image":
            try:
                image_actions.append({
                    "action": "add_inspo",
                    "image_id": int(block.input.get("image_id")),
                    "caption": (block.input.get("caption") or "").strip() or None,
                })
            except Exception as e:  # noqa: BLE001
                reply_text_parts.append(
                    f"\n\n_(add_inspo_image rejected: {e})_"
                )
        elif block.type == "tool_use" and block.name == "set_anatomy_diagram":
            try:
                image_actions.append({
                    "action": "set_anatomy",
                    "image_id": int(block.input.get("image_id")),
                    "caption": (block.input.get("caption") or "").strip() or None,
                })
            except Exception as e:  # noqa: BLE001
                reply_text_parts.append(
                    f"\n\n_(set_anatomy_diagram rejected: {e})_"
                )
        elif block.type == "tool_use" and block.name == "add_assembly_image":
            try:
                raw_step = block.input.get("step_index")
                image_actions.append({
                    "action": "add_assembly",
                    "image_id": int(block.input.get("image_id")),
                    "step_index": int(raw_step) if raw_step is not None else None,
                    "caption": (block.input.get("caption") or "").strip() or None,
                })
            except Exception as e:  # noqa: BLE001
                reply_text_parts.append(
                    f"\n\n_(add_assembly_image rejected: {e})_"
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
    return reply_text, updated, new_history, regen_requests, image_actions
