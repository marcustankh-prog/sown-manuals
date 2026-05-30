"""Streamlit UI for the Beaded Flower Manual generator.

Run with:
    streamlit run src/app.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Allow running as `streamlit run src/app.py` (no package context)
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from src.models import Flower  # noqa: E402
from src import pdf_generator  # noqa: E402
from src import library  # noqa: E402

load_dotenv(ROOT.parent / ".env", override=True)

# In hosted environments (Streamlit Community Cloud) there is no .env file;
# instead, secrets are provided via st.secrets. Bridge them into os.environ
# so the rest of the codebase (which reads os.getenv) works unchanged.
try:
    for _key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_IMAGE_MODEL",
        "ANTHROPIC_MODEL",
        "AI_PROVIDER",
    ):
        if not os.getenv(_key) and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # noqa: BLE001
    # st.secrets raises if no secrets.toml exists locally — that's fine.
    pass

UPLOADS = ROOT / "data" / "uploads"
OUTPUT = ROOT.parent / "output"
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
SAMPLE_DAISY = ROOT / "data" / "sample_daisy.json"


# ---------- Session helpers --------------------------------------------------

def _load_default_flower() -> Flower:
    if SAMPLE_DAISY.exists():
        return Flower.model_validate_json(SAMPLE_DAISY.read_text(encoding="utf-8"))
    return Flower(name="Flower", title="French Beaded Flower")


def _ensure_state() -> Flower:
    f = st.session_state.get("flower")
    if f is None:
        st.session_state.flower = _load_default_flower()
        return st.session_state.flower
    # If the model class has gained new fields since this object was created
    # (Streamlit hot-reload), round-trip through validate to fill defaults.
    try:
        st.session_state.flower = Flower.model_validate(f.model_dump())
    except Exception:  # noqa: BLE001
        st.session_state.flower = _load_default_flower()
    return st.session_state.flower


def _save_uploaded(file, name_hint: str) -> Path:
    safe = "".join(c for c in name_hint if c.isalnum() or c in "._-") or "img"
    target = UPLOADS / f"{datetime.now():%Y%m%d_%H%M%S}_{safe}_{file.name}"
    target.write_bytes(file.getbuffer())
    return target


def _img_src(path_or_uri: str) -> str:
    """Streamlit's st.image can't open file:// URIs; convert to local path."""
    if path_or_uri.startswith("file:"):
        from urllib.parse import urlparse, unquote
        return unquote(urlparse(path_or_uri).path).lstrip("/")
    return path_or_uri


# ---------- UI ---------------------------------------------------------------

st.set_page_config(page_title="SOWN — Beaded Flower Manual", page_icon="🌿", layout="wide")

# === Brand styling (Sown.objects identity) ===================================

_APP_CSS = (ROOT / "static" / "app.css").read_text(encoding="utf-8")
st.html(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap" rel="stylesheet">'
    f"<style>{_APP_CSS}</style>"
)

flower = _ensure_state()

# === Top bar: title + settings popover ======================================

provider = os.getenv("AI_PROVIDER", "anthropic").lower()
env_key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
has_openai = bool(os.getenv("OPENAI_API_KEY"))

bar_l, bar_r = st.columns([0.85, 0.15])
with bar_l:
    st.html(
        '<div class="sown-header">'
        '<svg class="sown-mark" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="28" cy="28" r="20" stroke="#C4CABD" stroke-width="0.8" fill="none"/>'
        '<ellipse cx="28" cy="18" rx="5" ry="9" fill="#8A9180" opacity="0.55"/>'
        '<ellipse cx="28" cy="18" rx="5" ry="9" fill="#8A9180" opacity="0.55" transform="rotate(60 28 28)"/>'
        '<ellipse cx="28" cy="18" rx="5" ry="9" fill="#8A9180" opacity="0.55" transform="rotate(120 28 28)"/>'
        '<ellipse cx="28" cy="18" rx="5" ry="9" fill="#8A9180" opacity="0.55" transform="rotate(180 28 28)"/>'
        '<ellipse cx="28" cy="18" rx="5" ry="9" fill="#8A9180" opacity="0.55" transform="rotate(240 28 28)"/>'
        '<ellipse cx="28" cy="18" rx="5" ry="9" fill="#8A9180" opacity="0.55" transform="rotate(300 28 28)"/>'
        '<circle cx="28" cy="28" r="3.5" fill="#5C6652"/>'
        '</svg>'
        '<div class="sown-text">'
        '<h1 class="sown-wordmark">SOWN</h1>'
        '<div class="sown-tagline">Beaded Flowers &amp; Botanical Artistry</div>'
        '<div class="sown-divider"></div>'
        '<p class="sown-subtitle">A quiet studio for crafting beaded-flower manuals — one bead, one petal at a time.</p>'
        '</div>'
        '</div>'
    )
with bar_r:
    with st.popover("Settings", use_container_width=True):
        st.caption("Illustration counts (used when generating images).")
        n_per_comp = st.slider("Images per component", 0, 3, 2, key="cfg_per_comp")
        n_assembly = st.slider("Assembly images", 0, 3, 2, key="cfg_assembly")
        n_inspo = st.slider("Inspo images", 0, 6, 4, key="cfg_inspo")
        st.divider()
        if st.button("Reset to sample Daisy", use_container_width=True):
            st.session_state.flower = _load_default_flower()
            st.session_state.chat_messages = []
            st.session_state.chat_history = []
            st.rerun()
        if st.button("Start with blank flower", use_container_width=True):
            st.session_state.flower = Flower(
                name="Flower", title="French Beaded Flower"
            )
            st.session_state.chat_messages = []
            st.session_state.chat_history = []
            st.rerun()
        if st.button("Clear chat history", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.chat_history = []
            st.rerun()
        st.divider()
        st.download_button(
            "Download JSON spec",
            data=flower.model_dump_json(indent=2),
            file_name=f"{flower.name.lower().replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
        )
        spec = st.file_uploader("Restore from JSON spec", type=["json"], key="spec")
        if spec is not None:
            try:
                data = json.loads(spec.read())
                st.session_state.flower = Flower.model_validate(data)
                st.success("Loaded.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not load spec: {e}")

if not has_anthropic:
    st.warning(
        "Add `ANTHROPIC_API_KEY` to `.env` (or Streamlit secrets) to enable AI drafting and chat editing."
    )

IMAGES_ENABLED = os.getenv("IMAGES_ENABLED", "false").lower() in ("1", "true", "yes", "on")

if not IMAGES_ENABLED:
    st.info(
        "💸 Illustration generation is currently **disabled** to save API costs while "
        "we refine the written instructions. The manual will render as text only. "
        "Set `IMAGES_ENABLED=true` in `.env` or Streamlit secrets to re-enable."
    )
elif not has_openai:
    st.warning(
        "Add `OPENAI_API_KEY` to `.env` (or Streamlit secrets) to enable "
        "illustration & inspo-photo generation. Without it, all image "
        "generation is skipped \u2014 the manual will render with text only."
    )

# === Main area: chat (left) + live preview (right) ==========================

# Initialise chat state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # for display: [{role, content}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # for the API: same shape

col_chat, col_preview = st.columns([0.42, 0.58], gap="large")

with col_chat:
    # ---- Saved manuals library -----------------------------------------
    _entries = library.list_entries()
    with st.expander(
        f"Saved manuals ({len(_entries)})",
        expanded=bool(_entries) and not flower.components,
    ):
        if not _entries:
            st.caption(
                "No saved manuals yet. Drafts auto-save here after generation "
                "and after each chat edit, so you can pick up where you left off."
            )
        else:
            st.caption(
                "Click **Load** to continue editing a previous manual. The chat "
                "editor still works on loaded manuals \u2014 ask Claude to refine "
                "any section."
            )
            for ent in _entries:
                lib_l, lib_m, lib_r = st.columns([0.5, 0.25, 0.25])
                with lib_l:
                    st.markdown(
                        f"**{ent['name']}**  \n"
                        f"<span style='color:#7A7B75;font-size:0.85em'>"
                        f"{ent['components']} components \u00b7 {ent['modified']}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                with lib_m:
                    if st.button("Load", key=f"lib_load_{ent['slug']}", use_container_width=True):
                        try:
                            loaded = library.load(ent["slug"])
                            st.session_state.flower = loaded
                            st.session_state.chat_messages = []
                            st.session_state.chat_history = []
                            st.session_state.save_name_input = (
                                loaded.library_label or loaded.name
                            )
                            st.success(f"Loaded \u201c{ent['name']}\u201d.")
                            st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Could not load: {e}")
                with lib_r:
                    if st.button("Delete", key=f"lib_del_{ent['slug']}", use_container_width=True):
                        library.delete(ent["slug"])
                        st.rerun()
        if flower.components:
            st.divider()
            default_save_name = st.session_state.get(
                "save_name_input", flower.library_label or flower.name
            )
            save_name = st.text_input(
                "Save as",
                value=default_save_name,
                key="save_name_input",
                help=(
                    "File name for this manual in your library. Use this to "
                    "keep multiple variants of the same flower (e.g. "
                    "“peony — pink v2”). Saving under an existing "
                    "name overwrites it."
                ),
            )
            slug_preview = library.slugify(save_name)
            already = library.exists(slug_preview)
            st.caption(
                f"Saved as `{slug_preview}.json`"
                + (" — will **overwrite** existing entry." if already else ".")
            )
            if st.button(
                "\U0001F4BE Save current manual",
                use_container_width=True,
                disabled=not save_name.strip(),
            ):
                library.save(flower, save_name=save_name)
                st.success(f"Saved as “{save_name}”.")
                st.rerun()

    # ---- Photo + Generate (chat-first onboarding) -----------------------
    with st.expander(
        "Start from a photo",
        expanded=not st.session_state.chat_messages and not flower.components,
    ):
        uploaded = st.file_uploader(
            "Drop up to 5 flower photos here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            accept_multiple_files=True,
            help=(
                "Upload 1\u20135 reference photos of the same flower "
                "(different angles, close-ups, leaves, back of a petal). "
                "The first photo becomes the cover; the rest are "
                "cross-referenced for accuracy."
            ),
        )
        if uploaded and len(uploaded) > 5:
            st.warning(
                f"You uploaded {len(uploaded)} photos \u2014 only the first 5 "
                "will be used."
            )
            uploaded = uploaded[:5]
        name_hint = st.text_input(
            "What flower is this?", value=flower.name, key="name_hint"
        )
        lang_choice = st.radio(
            "Manual language",
            options=["English", "\ud55c\uad6d\uc5b4 (Korean)"],
            horizontal=True,
            key="lang_choice",
        )
        lang_code = "ko" if lang_choice.startswith("\ud55c") else "en"
        recreate_mode = st.checkbox(
            "\U0001FAA1 This is already a beaded flower \u2014 recreate it exactly",
            value=False,
            key="recreate_mode",
            help=(
                "Tick this if your photo is a finished beaded piece (not a "
                "living plant). The pattern will copy its layer counts, "
                "petal shapes, and bead colours rather than improvise from a "
                "real flower."
            ),
        )
        gen_mode = "recreate" if recreate_mode else "plant"
        if uploaded:
            cols = st.columns(min(len(uploaded), 4))
            for i, f in enumerate(uploaded):
                with cols[i % len(cols)]:
                    st.image(f, use_container_width=True,
                             caption="primary" if i == 0 else f"ref {i}")
        gen_disabled = not (uploaded and has_anthropic)
        gen_label = (
            "Generate draft & illustrations"
            if (IMAGES_ENABLED and has_openai)
            else "Generate draft (text only)"
        )
        if st.button(
            gen_label,
            disabled=gen_disabled,
            use_container_width=True,
            type="primary",
        ):
            with st.spinner(
                f"Analyzing {len(uploaded)} photo{'s' if len(uploaded) != 1 else ''}\u2026"
            ):
                from src import ai_analyzer

                saved_paths = [_save_uploaded(f, name_hint) for f in uploaded]
                try:
                    draft = ai_analyzer.analyze_photo(
                        saved_paths,
                        hint_name=name_hint or None,
                        language=lang_code,
                        mode=gen_mode,
                    )
                    draft.hero_image = saved_paths[0].as_uri()
                    st.session_state.flower = draft
                    library.save(draft)
                except Exception as e:  # noqa: BLE001
                    st.error(f"AI text draft failed: {e}")
                    st.stop()

            if IMAGES_ENABLED and has_openai:
                from src import image_generator

                out_dir = (
                    UPLOADS
                    / f"gen_{datetime.now():%Y%m%d_%H%M%S}_{draft.name.lower()}"
                )
                bar = st.progress(0.0, text="Generating illustrations…")

                def _progress(label: str, frac: float):
                    bar.progress(min(frac, 1.0), text=label)

                try:
                    image_generator.generate_for_flower(
                        draft,
                        out_dir=out_dir,
                        images_per_component=n_per_comp,
                        assembly_images=n_assembly,
                        inspo_images=n_inspo,
                        progress=_progress,
                    )
                    bar.progress(1.0, text="Done.")
                    st.session_state.flower = draft
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Text draft saved, but illustrations failed: {e}")
                if image_generator.LAST_ERRORS:
                    with st.expander(
                        f"⚠️ {len(image_generator.LAST_ERRORS)} image(s) failed — details",
                        expanded=True,
                    ):
                        for err in image_generator.LAST_ERRORS:
                            st.code(err, language="text")
            else:
                if not IMAGES_ENABLED:
                    st.info(
                        "Text draft generated. Illustration generation is "
                        "disabled (set `IMAGES_ENABLED=true` to enable)."
                    )
                else:
                    st.info(
                        "Text draft generated. Add `OPENAI_API_KEY` to enable "
                        "illustration generation."
                    )
            st.rerun()

        if flower.components and IMAGES_ENABLED:
            if st.button(
                "Regenerate illustrations",
                use_container_width=True,
                disabled=not has_openai,
                help=(
                    "Uses OpenAI's image API (~$0.04 per image)."
                    if has_openai
                    else "Add OPENAI_API_KEY to enable."
                ),
            ):
                from src import image_generator

                out_dir = (
                    UPLOADS
                    / f"gen_{datetime.now():%Y%m%d_%H%M%S}_{flower.name.lower()}"
                )
                bar = st.progress(0.0, text="Starting…")

                def _progress(label: str, frac: float):
                    bar.progress(min(frac, 1.0), text=label)

                try:
                    image_generator.generate_for_flower(
                        flower,
                        out_dir=out_dir,
                        images_per_component=n_per_comp,
                        assembly_images=n_assembly,
                        inspo_images=n_inspo,
                        progress=_progress,
                    )
                    st.session_state.flower = flower
                    bar.progress(1.0, text="Done.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Image generation failed: {e}")
                if image_generator.LAST_ERRORS:
                    with st.expander(
                        f"⚠️ {len(image_generator.LAST_ERRORS)} image(s) failed — details",
                        expanded=True,
                    ):
                        for err in image_generator.LAST_ERRORS:
                            st.code(err, language="text")
                else:
                    st.rerun()

    # ---- Source-photo classifier warning --------------------------------
    _src_kind = getattr(flower, "source_kind", None)
    _src_mode = getattr(flower, "source_mode", "plant")
    _src_reason = getattr(flower, "source_classifier_reason", "") or ""
    if _src_kind == "beaded" and _src_mode == "plant":
        st.warning(
            "🪡 This photo looks like an **already-beaded flower**, but the "
            "manual was generated in *plant mode* — petal counts and anatomy "
            "may not match the piece in the photo."
            + (f"\n\n_Classifier note: {_src_reason}_" if _src_reason else "")
        )
        if has_anthropic and st.button(
            "Regenerate as a recreation of this beaded piece",
            key="regen_recreate",
            use_container_width=True,
        ):
            from urllib.parse import urlparse, unquote
            from src import ai_analyzer

            hero = flower.hero_image or ""
            src = (
                Path(unquote(urlparse(hero).path).lstrip("/"))
                if hero.startswith("file:")
                else Path(hero) if hero else None
            )
            if src and src.exists():
                with st.spinner("Regenerating in recreation mode…"):
                    try:
                        draft = ai_analyzer.analyze_photo(
                            src,
                            hint_name=flower.name,
                            language=flower.language,
                            mode="recreate",
                        )
                        draft.hero_image = src.as_uri()
                        st.session_state.flower = draft
                        library.save(draft)
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Regeneration failed: {e}")
            else:
                st.error("Original photo file is no longer available; please re-upload.")
    elif _src_kind == "illustration" and _src_mode == "plant":
        st.info(
            "🎨 This photo looks like an illustration or stylised rendering "
            "rather than a real-plant photograph. The pattern was generated "
            "in plant mode — anatomy may be approximate."
            + (f"\n\n_Classifier note: {_src_reason}_" if _src_reason else "")
        )

    # ---- Chat ------------------------------------------------------------
    chat_box = st.container(height=480)
    with chat_box:
        if not st.session_state.chat_messages:
            st.info(
                "Drop a flower photo above and click *Generate*, "
                "or just start chatting to build a manual from scratch. "
                "Try: *'make the outer petals 16 instead of 12'* or "
                "*'rewrite the intro warmer'*."
            )
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_msg = st.chat_input(
        "Ask a question or request a change…",
        disabled=not has_anthropic,
    )
    if user_msg:
        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        with st.spinner("Claude is thinking…"):
            try:
                from src import chat_editor

                reply, updated, new_hist, regen_requests = chat_editor.chat(
                    flower,
                    st.session_state.chat_history,
                    user_msg,
                )
                st.session_state.chat_history = new_hist
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": reply}
                )
                if updated is not None:
                    st.session_state.flower = updated
                    flower = updated
                    library.save(updated)
            except Exception as e:  # noqa: BLE001
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": f"⚠️ Error: {e}"}
                )
                regen_requests = []

        # Run any image-regeneration requests Claude asked for.
        if regen_requests and IMAGES_ENABLED and has_openai:
            from src import image_generator

            for req in regen_requests:
                scope = req["scope"]
                scopes = (
                    image_generator.SCOPES_ALL
                    if scope == "all"
                    else (scope,)
                )
                out_dir = (
                    UPLOADS
                    / f"gen_{datetime.now():%Y%m%d_%H%M%S}_{flower.name.lower()}_{scope}"
                )
                bar = st.progress(0.0, text=f"Regenerating {scope}…")

                def _progress(label: str, frac: float):
                    bar.progress(min(frac, 1.0), text=label)

                try:
                    image_generator.generate_for_flower(
                        flower,
                        out_dir=out_dir,
                        images_per_component=n_per_comp,
                        assembly_images=n_assembly,
                        inspo_images=n_inspo,
                        progress=_progress,
                        scopes=scopes,
                        brief_override=req.get("brief_override"),
                    )
                    bar.progress(1.0, text=f"{scope.title()} regenerated.")
                    st.session_state.flower = flower
                except Exception as e:  # noqa: BLE001
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": f"⚠️ {scope.title()} regeneration failed: {e}",
                        }
                    )
        elif regen_requests:
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "⚠️ I asked to regenerate illustrations, but "
                        "image generation is currently disabled "
                        "(or `OPENAI_API_KEY` is not configured)."
                    ),
                }
            )
        st.rerun()

with col_preview:
    pcol1, pcol2 = st.columns([0.7, 0.3])
    with pcol1:
        st.markdown("**Live preview**")
    with pcol2:
        if st.button("Generate PDF", use_container_width=True, type="primary"):
            with st.spinner("Rendering PDF…"):
                try:
                    out = OUTPUT / f"{flower.name.lower().replace(' ', '_')}_manual.pdf"
                    pdf_generator.to_pdf(flower, out)
                    st.session_state.last_pdf = out
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))

    html = pdf_generator.render_html(flower, embed_css=True)
    st.components.v1.html(html, height=900, scrolling=True)

    last_pdf = st.session_state.get("last_pdf")
    dl_l, dl_r = st.columns(2)
    with dl_l:
        st.download_button(
            "Download HTML",
            data=html,
            file_name=f"{flower.name.lower().replace(' ', '_')}_manual.html",
            mime="text/html",
            use_container_width=True,
        )
    with dl_r:
        if last_pdf and Path(last_pdf).exists():
            with open(last_pdf, "rb") as fh:
                st.download_button(
                    "Download PDF",
                    data=fh.read(),
                    file_name=Path(last_pdf).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.caption("Click *Generate PDF* to enable download.")

st.session_state.flower = flower
