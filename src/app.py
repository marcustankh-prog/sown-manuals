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

from src.models import Flower, ComponentImage  # noqa: E402
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
    """Empty flower for first-load welcome state."""
    return Flower(name="Flower", title="French Beaded Flower")


def _load_sample_flower() -> Flower:
    if SAMPLE_DAISY.exists():
        return Flower.model_validate_json(SAMPLE_DAISY.read_text(encoding="utf-8"))
    return _load_default_flower()


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


# Per-target image-processing presets used by both the manual uploaders
# and the chat upload handler so every uploaded photo is cropped/squared.
# bg="remove" is required for auto-crop to find the subject in a busy photo.
_IMAGE_PRESETS = {
    "hero":           {"mode": "photo",  "bg": "remove"},
    "anatomy":        {"mode": "auto",   "bg": "remove"},
    "assembly":       {"mode": "auto",   "bg": "remove"},
    "inspo":          {"mode": "photo",  "bg": "remove"},
    "component_step": {"mode": "auto",   "bg": "remove"},
}


def _clean_to(target: str, src_path: Path, dest: Path) -> Path:
    """Run clean_image with the target's preset; fall back to a raw copy."""
    from src import image_processing
    preset = _IMAGE_PRESETS.get(target, {"mode": "auto", "bg": "remove"})
    try:
        image_processing.clean_image(src_path, dest, **preset)
    except Exception as e:  # noqa: BLE001
        # Worst case: re-encode the source as PNG so the manual still
        # gets a valid image, and surface the error so we can debug.
        try:
            from PIL import Image, ImageOps
            img = ImageOps.exif_transpose(Image.open(src_path)).convert("RGB")
            img.save(dest, format="PNG", optimize=True)
        except Exception:  # noqa: BLE001
            if src_path.resolve() != dest.resolve():
                dest.write_bytes(src_path.read_bytes())
        try:
            st.warning(f"Image cleaning fell back to raw copy: {e}")
        except Exception:  # noqa: BLE001
            pass
    return dest


# ---------- UI ---------------------------------------------------------------

st.set_page_config(page_title="SOWN — Beaded Flower Manual", page_icon="🌿", layout="wide")

# === Brand styling (Sown.objects identity) ===================================

_APP_CSS = (ROOT / "static" / "app.css").read_text(encoding="utf-8")
# Inject Google Fonts <link> tags and the brand <style> block separately —
# combining them in one markdown call caused Streamlit to render the CSS
# as visible text instead of applying it.
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown(f"<style>{_APP_CSS}</style>", unsafe_allow_html=True)

flower = _ensure_state()

# === Top bar: title + settings popover ======================================

provider = os.getenv("AI_PROVIDER", "anthropic").lower()
env_key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
has_openai = bool(os.getenv("OPENAI_API_KEY"))

bar_l, bar_r = st.columns([0.85, 0.15])
with bar_l:
    st.markdown(
        '<div class="sown-header">'
        '<div class="sown-text">'
        '<h1 class="sown-wordmark">SOWN</h1>'
        '<div class="sown-tagline">Beaded Flowers &amp; Botanical Artistry</div>'
        '<div class="sown-divider"></div>'
        '<p class="sown-subtitle">A quiet studio for crafting beaded-flower manuals — one bead, one petal at a time.</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with bar_r:
    with st.popover("Settings", use_container_width=True):
        st.caption("Illustration counts (used when generating images).")
        n_per_comp = st.slider("Images per component", 0, 3, 2, key="cfg_per_comp")
        n_assembly = st.slider("Assembly images", 0, 3, 2, key="cfg_assembly")
        n_inspo = st.slider("Inspo images", 0, 6, 4, key="cfg_inspo")
        st.divider()
        if st.button("Reset to sample Daisy", use_container_width=True):
            st.session_state.flower = _load_sample_flower()
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

if IMAGES_ENABLED and not has_openai:
    st.warning(
        "Add `OPENAI_API_KEY` to `.env` (or Streamlit secrets) to enable "
        "illustration & inspo-photo generation. Without it, all image "
        "generation is skipped \u2014 the manual will render with text only."
    )

# === Main area =============================================================

# Initialise chat state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # for display: [{role, content}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # for the API: same shape

# Welcome mode (no manual yet) → single column with just Saved manuals + Start
# from a photo. Editing mode (a draft exists) → two-column chat + preview.
HAS_MANUAL = bool(flower.components)

if HAS_MANUAL:
    col_chat, col_preview = st.columns([0.42, 0.58], gap="large")
    _left_container = col_chat
else:
    # Welcome screen uses the full main column so panels resize fluidly.
    _left_container = st.container()
    with _left_container:
        st.markdown(
            "<div style='font-family:Cormorant Garamond,serif;"
            "font-style:italic;color:#5C6652;font-size:1.15rem;"
            "margin:1.5rem 0 1.25rem 0;line-height:1.5;'>"
            "Begin a new beaded-flower manual — open a saved draft below, "
            "or upload reference photographs to draft one from scratch."
            "</div>",
            unsafe_allow_html=True,
        )

with _left_container:
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
                                getattr(loaded, "library_label", None)
                                or loaded.name
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
                "save_name_input",
                getattr(flower, "library_label", None) or flower.name,
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

    # Everything from here on is editing-mode only — only show once the
    # user has a generated draft (components exist).
    if not HAS_MANUAL:
        st.stop()

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

    # ---- Step illustrations (manual upload + deterministic cleanup) ------
    if flower.components:
        _total_steps = sum(len(c.paragraphs) for c in flower.components)
        _illustrated = sum(
            1
            for c in flower.components
            for img in c.images
            if img.step_index is not None
        )
        with st.expander(
            f"📐 Step illustrations ({_illustrated} / {_total_steps})",
            expanded=False,
        ):
            st.markdown(
                "<div style='font-family:Jost,sans-serif;font-size:0.86rem;"
                "color:#2E2A22;line-height:1.45;margin:0 0 0.6rem 0;'>"
                "Upload a sketch or photo for each step. The app auto-detects "
                "whether it's hand-drawn or a photo, removes the background, "
                "squares the crop and inserts it. <em>Auto</em> picks line-art "
                "for sketches and photo style for photos — override per row "
                "if you'd rather lock one or the other."
                "</div>",
                unsafe_allow_html=True,
            )
            _slug = library.slugify(
                getattr(flower, "library_label", None) or flower.name
            )
            _steps_dir = UPLOADS / "steps" / _slug

            for c_idx, comp in enumerate(flower.components):
                st.markdown(f"**{comp.heading.rstrip(':')}**")
                if not comp.paragraphs:
                    st.caption("_(no steps yet)_")
                    continue
                for s_idx, para in enumerate(comp.paragraphs):
                    kp = f"stepimg_{c_idx}_{s_idx}"
                    cur = next(
                        (img for img in comp.images if img.step_index == s_idx),
                        None,
                    )
                    # Step description on its own full-width row.
                    excerpt = para[:170] + ("…" if len(para) > 170 else "")
                    st.markdown(
                        f"<div style='font-family:Jost,sans-serif;"
                        f"font-size:0.86rem;color:#2A2B27;line-height:1.5;'>"
                        f"<span style='color:#7A7B75;letter-spacing:0.1em;"
                        f"text-transform:uppercase;font-size:0.72rem;'>"
                        f"Step {s_idx + 1}</span><br>"
                        f"{excerpt}</div>",
                        unsafe_allow_html=True,
                    )
                    # Thumbnail / no-image placeholder + uploader on row 2.
                    r1, r2 = st.columns([0.22, 0.78])
                    with r1:
                        if cur and cur.path:
                            try:
                                st.image(cur.path, use_container_width=True)
                            except Exception:  # noqa: BLE001
                                st.caption("_(preview n/a)_")
                            if st.button("Remove", key=f"{kp}_rm",
                                         use_container_width=True):
                                comp.images = [
                                    i for i in comp.images
                                    if i.step_index != s_idx
                                ]
                                library.save(flower)
                                st.rerun()
                        else:
                            st.caption("_no image yet_")
                    with r2:
                        upl = st.file_uploader(
                            "Upload photo or sketch",
                            type=["jpg", "jpeg", "png", "webp"],
                            key=f"{kp}_up",
                            label_visibility="collapsed",
                        )
                    # Controls on row 3 — wide enough that nothing truncates.
                    c1, c2, c3 = st.columns([0.36, 0.36, 0.28])
                    with c1:
                        style = st.selectbox(
                            "Style",
                            options=["auto", "line_art", "photo"],
                            format_func=lambda v: {
                                "auto": "Auto-detect",
                                "line_art": "Line drawing",
                                "photo": "Photo",
                            }[v],
                            index=0,
                            key=f"{kp}_style",
                            help=(
                                "How to render the image. "
                                "Auto picks line drawing for sketches and "
                                "photo for photographs. Line drawing "
                                "converts to black outlines on white."
                            ),
                        )
                    with c2:
                        bg = st.selectbox(
                            "Background",
                            options=["remove", "keep"],
                            format_func=lambda v: {
                                "remove": "Remove",
                                "keep": "Keep",
                            }[v],
                            index=0,
                            key=f"{kp}_bg",
                            help=(
                                "Remove cuts out the subject (best for "
                                "clean step diagrams). Keep leaves the "
                                "original background untouched."
                            ),
                        )
                    with c3:
                        st.markdown(
                            "<div style='height:1.85rem'></div>",
                            unsafe_allow_html=True,
                        )
                        if upl is not None and st.button(
                            "Clean & attach", key=f"{kp}_btn",
                            use_container_width=True,
                        ):
                            from src import image_processing

                            _steps_dir.mkdir(parents=True, exist_ok=True)
                            raw = _steps_dir / (
                                f"raw_{c_idx}_{s_idx}{Path(upl.name).suffix}"
                            )
                            raw.write_bytes(upl.getbuffer())
                            out = _steps_dir / f"step_{c_idx}_{s_idx}.png"
                            try:
                                with st.spinner("Cleaning…"):
                                    image_processing.clean_image(
                                        raw, out, mode=style, bg=bg
                                    )
                                comp.images = [
                                    i for i in comp.images
                                    if i.step_index != s_idx
                                ]
                                comp.images.append(ComponentImage(
                                    path=out.as_uri(),
                                    step_index=s_idx,
                                ))
                                library.save(flower)
                                st.rerun()
                            except Exception as e:  # noqa: BLE001
                                st.error(f"Cleanup failed: {e}")
                    st.divider()

    # ---- Other manual image uploads (hero / anatomy / assembly / inspo) --
    with st.expander("🖼️ Cover, anatomy, assembly & inspo photos", expanded=False):
        st.markdown(
            "<div style='font-family:Jost,sans-serif;font-size:0.86rem;"
            "color:#2E2A22;line-height:1.45;margin:0 0 0.6rem 0;'>"
            "Upload photos for the cover, anatomy diagram, final-assembly "
            "section, and back-of-manual inspo gallery. Files are saved as-is "
            "(no background removal)."
            "</div>",
            unsafe_allow_html=True,
        )
        _slug_misc = library.slugify(
            getattr(flower, "library_label", None) or flower.name
        )
        _misc_dir = UPLOADS / "manual" / _slug_misc
        from src.models import AssemblySection

        # --- Cover / hero ---
        st.markdown("**Cover photo**")
        hr1, hr2 = st.columns([0.22, 0.78])
        with hr1:
            if flower.hero_image:
                try:
                    st.image(flower.hero_image, use_container_width=True)
                except Exception:  # noqa: BLE001
                    st.caption("_(preview n/a)_")
                if st.button("Remove", key="hero_rm", use_container_width=True):
                    flower.hero_image = None
                    library.save(flower)
                    st.rerun()
            else:
                st.caption("_no image yet_")
        with hr2:
            up_hero = st.file_uploader(
                "Upload cover photo",
                type=["jpg", "jpeg", "png", "webp"],
                key="hero_up",
                label_visibility="collapsed",
            )
            if up_hero is not None and st.button(
                "Set as cover", key="hero_btn", use_container_width=True
            ):
                _misc_dir.mkdir(parents=True, exist_ok=True)
                raw = _misc_dir / f"hero_raw{Path(up_hero.name).suffix.lower()}"
                raw.write_bytes(up_hero.getbuffer())
                out = _misc_dir / "hero.png"
                with st.spinner("Cleaning…"):
                    _clean_to("hero", raw, out)
                flower.hero_image = out.as_uri()
                library.save(flower)
                st.rerun()
        st.divider()

        # --- Anatomy diagram ---
        st.markdown("**Anatomy diagram**")
        ar1, ar2 = st.columns([0.22, 0.78])
        with ar1:
            if flower.anatomy_diagram and flower.anatomy_diagram.path:
                try:
                    st.image(flower.anatomy_diagram.path, use_container_width=True)
                except Exception:  # noqa: BLE001
                    st.caption("_(preview n/a)_")
                if st.button("Remove", key="anatomy_rm", use_container_width=True):
                    flower.anatomy_diagram = None
                    library.save(flower)
                    st.rerun()
            else:
                st.caption("_no image yet_")
        with ar2:
            up_anatomy = st.file_uploader(
                "Upload anatomy diagram",
                type=["jpg", "jpeg", "png", "webp"],
                key="anatomy_up",
                label_visibility="collapsed",
            )
            anatomy_caption = st.text_input(
                "Caption (optional)", key="anatomy_caption",
                value=(flower.anatomy_diagram.caption or "")
                if flower.anatomy_diagram else "",
            )
            if up_anatomy is not None and st.button(
                "Set as anatomy diagram", key="anatomy_btn",
                use_container_width=True,
            ):
                _misc_dir.mkdir(parents=True, exist_ok=True)
                raw = _misc_dir / f"anatomy_raw{Path(up_anatomy.name).suffix.lower()}"
                raw.write_bytes(up_anatomy.getbuffer())
                out = _misc_dir / "anatomy.png"
                with st.spinner("Cleaning…"):
                    _clean_to("anatomy", raw, out)
                flower.anatomy_diagram = ComponentImage(
                    path=out.as_uri(),
                    caption=anatomy_caption.strip() or None,
                )
                library.save(flower)
                st.rerun()
        st.divider()

        # --- Assembly images ---
        st.markdown("**Assembly section photos**")
        if flower.assembly and flower.assembly.images:
            cols_a = st.columns(min(len(flower.assembly.images), 4))
            for i, img in enumerate(flower.assembly.images):
                with cols_a[i % len(cols_a)]:
                    try:
                        st.image(img.path, use_container_width=True)
                    except Exception:  # noqa: BLE001
                        st.caption("_(preview n/a)_")
                    if st.button("Remove", key=f"asm_rm_{i}",
                                 use_container_width=True):
                        flower.assembly.images.pop(i)
                        library.save(flower)
                        st.rerun()
        else:
            st.caption("_no images yet_")
        up_asm = st.file_uploader(
            "Upload assembly photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="asm_up",
        )
        asm_caption = st.text_input("Caption (optional)", key="asm_caption")
        if up_asm is not None and st.button(
            "Add to assembly", key="asm_btn", use_container_width=True
        ):
            _misc_dir.mkdir(parents=True, exist_ok=True)
            existing = len(flower.assembly.images) if flower.assembly else 0
            raw = _misc_dir / (
                f"assembly_{existing}_raw{Path(up_asm.name).suffix.lower()}"
            )
            raw.write_bytes(up_asm.getbuffer())
            out = _misc_dir / f"assembly_{existing}.png"
            with st.spinner("Cleaning…"):
                _clean_to("assembly", raw, out)
            if flower.assembly is None:
                flower.assembly = AssemblySection()
            flower.assembly.images.append(ComponentImage(
                path=out.as_uri(),
                caption=asm_caption.strip() or None,
            ))
            library.save(flower)
            st.rerun()
        st.divider()

        # --- Inspo gallery ---
        st.markdown("**Inspo gallery**")
        if flower.inspo_images:
            cols_i = st.columns(min(len(flower.inspo_images), 4))
            for i, img in enumerate(flower.inspo_images):
                with cols_i[i % len(cols_i)]:
                    try:
                        st.image(img.path, use_container_width=True)
                    except Exception:  # noqa: BLE001
                        st.caption("_(preview n/a)_")
                    if st.button("Remove", key=f"inspo_rm_{i}",
                                 use_container_width=True):
                        flower.inspo_images.pop(i)
                        library.save(flower)
                        st.rerun()
        else:
            st.caption("_no images yet_")
        up_inspo = st.file_uploader(
            "Upload inspo photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="inspo_up",
        )
        inspo_caption = st.text_input("Caption (optional)", key="inspo_caption")
        if up_inspo is not None and st.button(
            "Add to inspo gallery", key="inspo_btn", use_container_width=True
        ):
            _misc_dir.mkdir(parents=True, exist_ok=True)
            existing = len(flower.inspo_images)
            raw = _misc_dir / (
                f"inspo_{existing}_raw{Path(up_inspo.name).suffix.lower()}"
            )
            raw.write_bytes(up_inspo.getbuffer())
            out = _misc_dir / f"inspo_{existing}.png"
            with st.spinner("Cleaning…"):
                _clean_to("inspo", raw, out)
            flower.inspo_images.append(ComponentImage(
                path=out.as_uri(),
                caption=inspo_caption.strip() or None,
            ))
            library.save(flower)
            st.rerun()

    # ---- Chat ------------------------------------------------------------
    st.markdown('<div class="sown-chat-label">Conversation</div>', unsafe_allow_html=True)
    chat_box = st.container(height=480, border=True, key="sown_chat_panel")
    with chat_box:
        if not st.session_state.chat_messages:
            st.markdown(
                '<div class="sown-chat-empty">Ask a question or request '
                'a change to begin.</div>',
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input(
        "Ask a question, attach a photo, or request a change…",
        disabled=not has_anthropic,
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "webp"],
    )
    if user_input:
        # `accept_file` makes chat_input return a ChatInputValue, not a str.
        if isinstance(user_input, str):
            user_msg = user_input
            uploaded_files = []
        else:
            user_msg = (user_input.text or "").strip()
            uploaded_files = list(user_input.files or [])

        # Persist + clean any attached images right now so we can pass paths
        # to chat_editor and reuse them for tool-driven attachments.
        attached: list[dict] = []
        if uploaded_files:
            import base64
            from src import image_processing

            chat_dir = UPLOADS / "chat" / datetime.now().strftime("%Y%m%d_%H%M%S")
            chat_dir.mkdir(parents=True, exist_ok=True)
            for i, uf in enumerate(uploaded_files, start=1):
                raw = chat_dir / f"raw_{i}{Path(uf.name).suffix}"
                raw.write_bytes(uf.getbuffer())
                cleaned = chat_dir / f"img_{i}.png"
                try:
                    image_processing.clean_image(
                        raw, cleaned, mode="auto", bg="remove"
                    )
                except Exception:  # noqa: BLE001
                    cleaned = raw  # fall back to the original
                b64 = base64.standard_b64encode(cleaned.read_bytes()).decode()
                attached.append({
                    "id": i,
                    "path": cleaned,
                    "mime": "image/png",
                    "b64": b64,
                })

        display_user = user_msg
        if attached:
            display_user += (
                f"\n\n_({len(attached)} image(s) attached)_"
                if user_msg else f"_(attached {len(attached)} image(s))_"
            )
        st.session_state.chat_messages.append(
            {"role": "user", "content": display_user or "_(image only)_"}
        )

        regen_requests: list = []
        image_actions: list = []
        # Render the just-added user turn and a live "thinking" assistant
        # bubble inside the chat panel so progress stays in-flow.
        with chat_box:
            with st.chat_message("user"):
                st.markdown(display_user or "_(image only)_")
            with st.chat_message("assistant"):
                thinking_slot = st.empty()
                thinking_slot.markdown(
                    '<div class="sown-thinking">'
                    '<span class="sown-thinking-dot"></span>'
                    '<span class="sown-thinking-dot"></span>'
                    '<span class="sown-thinking-dot"></span>'
                    '<span class="sown-thinking-text">Sown is thinking…</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        try:
            from src import chat_editor

            reply, updated, new_hist, regen_requests, image_actions = chat_editor.chat(
                flower,
                st.session_state.chat_history,
                user_msg or "(image only)",
                attached_images=attached or None,
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

        # Apply image-attachment actions Claude requested.
        if image_actions:
            id_to_path = {a["id"]: a["path"] for a in (attached or [])}
            _slug_chat = library.slugify(
                getattr(flower, "library_label", None) or flower.name
            )
            _chat_clean_dir = UPLOADS / "manual" / _slug_chat
            _chat_clean_dir.mkdir(parents=True, exist_ok=True)

            def _process_chat_image(target: str, src: str, suffix: str = "") -> str:
                """Run the chat-attached file through clean_image and return file:// URI."""
                src_p = Path(src)
                stem = f"{target}{suffix}_{src_p.stem}"
                out = _chat_clean_dir / f"{stem}.png"
                _clean_to(target, src_p, out)
                return out.as_uri()

            for action in image_actions:
                act = action["action"]

                # Removal actions don't reference an attached image.
                if act == "remove_image":
                    target = action.get("target")
                    idx = action.get("index")
                    c_idx = action.get("component_index")
                    s_idx = action.get("step_index")
                    if target == "hero":
                        flower.hero_image = None
                    elif target == "anatomy":
                        flower.anatomy_diagram = None
                    elif target == "inspo":
                        if isinstance(idx, int) and 0 <= idx < len(flower.inspo_images):
                            flower.inspo_images.pop(idx)
                        else:
                            flower.inspo_images.clear()
                    elif target == "assembly" and flower.assembly:
                        if isinstance(idx, int) and 0 <= idx < len(flower.assembly.images):
                            flower.assembly.images.pop(idx)
                        else:
                            flower.assembly.images.clear()
                    elif target == "component_step":
                        if isinstance(c_idx, int) and 0 <= c_idx < len(flower.components):
                            comp = flower.components[c_idx]
                            if isinstance(s_idx, int):
                                comp.images = [
                                    i for i in comp.images if i.step_index != s_idx
                                ]
                            else:
                                comp.images.clear()
                    continue

                img_id = action.get("image_id")
                src_path = id_to_path.get(img_id)
                if not src_path or not Path(src_path).exists():
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Could not find attached image #{img_id}.",
                    })
                    continue
                if act == "set_hero":
                    flower.hero_image = _process_chat_image("hero", src_path)
                elif act == "add_inspo":
                    suffix = f"_{len(flower.inspo_images)}"
                    flower.inspo_images.append(ComponentImage(
                        path=_process_chat_image("inspo", src_path, suffix),
                        caption=action.get("caption"),
                    ))
                elif act == "set_anatomy":
                    flower.anatomy_diagram = ComponentImage(
                        path=_process_chat_image("anatomy", src_path),
                        caption=action.get("caption"),
                    )
                elif act == "add_assembly":
                    if flower.assembly is None:
                        from src.models import AssemblySection
                        flower.assembly = AssemblySection()
                    suffix = f"_{len(flower.assembly.images)}"
                    flower.assembly.images.append(ComponentImage(
                        path=_process_chat_image("assembly", src_path, suffix),
                        step_index=action.get("step_index"),
                        caption=action.get("caption"),
                    ))
                elif act == "attach_step":
                    c_idx = action["component_index"]
                    s_idx = action["step_index"]
                    if 0 <= c_idx < len(flower.components):
                        comp = flower.components[c_idx]
                        comp.images = [
                            i for i in comp.images if i.step_index != s_idx
                        ]
                        suffix = f"_c{c_idx}_s{s_idx if s_idx is not None else 'all'}"
                        comp.images.append(ComponentImage(
                            path=_process_chat_image("component_step", src_path, suffix),
                            step_index=s_idx,
                            caption=action.get("caption"),
                        ))
                    else:
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": (
                                f"⚠️ component_index {c_idx} out of range "
                                f"(have {len(flower.components)})."
                            ),
                        })
            st.session_state.flower = flower
            library.save(flower)

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
