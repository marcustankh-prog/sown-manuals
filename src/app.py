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

from src.models import (  # noqa: E402
    Anatomy,
    AssemblySection,
    Component,
    ComponentImage,
    Flower,
    Material,
    PetalLayer,
)
from src import pdf_generator  # noqa: E402

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

st.set_page_config(page_title="Beaded Flower Manual", page_icon="🌸", layout="wide")

flower = _ensure_state()

# === Top bar: title + settings popover ======================================

provider = os.getenv("AI_PROVIDER", "anthropic").lower()
env_key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
has_openai = bool(os.getenv("OPENAI_API_KEY"))

bar_l, bar_r = st.columns([0.85, 0.15])
with bar_l:
    st.title("🌸 Beaded Flower Manual")
    st.caption(
        "Upload a photo, get an AI draft, then chat to refine it. "
        "The preview on the right updates as you edit."
    )
with bar_r:
    with st.popover("⚙ Settings", use_container_width=True):
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
            "⬇ Download JSON spec",
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
        "Add `ANTHROPIC_API_KEY` to `.env` to enable AI drafting and chat editing."
    )

# === Main area: chat (left) + live preview (right) ==========================

# Initialise chat state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # for display: [{role, content}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # for the API: same shape

col_chat, col_preview = st.columns([0.42, 0.58], gap="large")

with col_chat:
    # ---- Photo + Generate (chat-first onboarding) -----------------------
    with st.expander(
        "📷 Start from a photo",
        expanded=not st.session_state.chat_messages and not flower.components,
    ):
        uploaded = st.file_uploader(
            "Drop a flower photo here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        name_hint = st.text_input(
            "What flower is this?", value=flower.name, key="name_hint"
        )
        if uploaded:
            st.image(uploaded, use_container_width=True)
        gen_disabled = not (uploaded and has_anthropic)
        if st.button(
            "✨ Generate draft + illustrations from photo",
            disabled=gen_disabled,
            use_container_width=True,
            type="primary",
        ):
            with st.spinner("Analyzing photo…"):
                from src import ai_analyzer

                saved = _save_uploaded(uploaded, name_hint)
                try:
                    draft = ai_analyzer.analyze_photo(
                        saved, hint_name=name_hint or None
                    )
                    draft.hero_image = saved.as_uri()
                    st.session_state.flower = draft
                except Exception as e:  # noqa: BLE001
                    st.error(f"AI text draft failed: {e}")
                    st.stop()

            if has_openai:
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
            st.rerun()

        if flower.components and has_openai:
            if st.button(
                "🎨 Regenerate illustrations",
                use_container_width=True,
                help="Uses OpenAI's image API (~$0.04 per image).",
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
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Image generation failed: {e}")

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
            except Exception as e:  # noqa: BLE001
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": f"⚠️ Error: {e}"}
                )
                regen_requests = []

        # Run any image-regeneration requests Claude asked for.
        if regen_requests and has_openai:
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
        elif regen_requests and not has_openai:
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "⚠️ I asked to regenerate illustrations, but "
                        "OPENAI_API_KEY is not configured."
                    ),
                }
            )
        st.rerun()

with col_preview:
    pcol1, pcol2 = st.columns([0.7, 0.3])
    with pcol1:
        st.markdown("**Live preview**")
    with pcol2:
        if st.button("🖨 Generate PDF", use_container_width=True, type="primary"):
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
            "⬇ Download HTML",
            data=html,
            file_name=f"{flower.name.lower().replace(' ', '_')}_manual.html",
            mime="text/html",
            use_container_width=True,
        )
    with dl_r:
        if last_pdf and Path(last_pdf).exists():
            with open(last_pdf, "rb") as fh:
                st.download_button(
                    "⬇ Download PDF",
                    data=fh.read(),
                    file_name=Path(last_pdf).name,
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.caption("Click *Generate PDF* to enable download.")

st.session_state.flower = flower
