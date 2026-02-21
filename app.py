"""
Text-to-Speech Web App
Upload a text file (TXT, PDF, MD, etc.) and generate AI-narrated audio.
"""

import io
import re
import tempfile
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from main import run_narrator_agent

# ─────────────────────────────────────────────
#  TEXT EXTRACTION
# ─────────────────────────────────────────────

SUPPORTED_TYPES = {
    ".txt": "Plain text",
    ".md": "Markdown",
    ".pdf": "PDF document",
    ".html": "HTML",
    ".rtf": "Rich text",
}


def extract_text_from_file(uploaded_file) -> str | None:
    """Extract raw text from an uploaded file based on its type."""
    suffix = Path(uploaded_file.name).suffix.lower()
    bytes_data = uploaded_file.read()

    try:
        if suffix == ".txt" or suffix == ".md" or suffix == ".html":
            return bytes_data.decode("utf-8", errors="replace")
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(bytes_data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".rtf":
            # Basic RTF: strip control words and keep text
            text = bytes_data.decode("utf-8", errors="replace")
            return re.sub(r"\\[a-z]+\d*\s?|[{}\\]", " ", text).strip()
    except Exception as e:
        st.error(f"Could not extract text from {uploaded_file.name}: {e}")
        return None

    st.warning(f"Unsupported file type: {suffix}")
    return None


# ─────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Text to Speech",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Text to Speech")
st.caption("Upload a text-based file and generate AI-narrated audio with multiple voices.")

st.divider()

# File upload
uploaded = st.file_uploader(
    "Choose a file",
    type=[ext.lstrip(".") for ext in SUPPORTED_TYPES],
    help=f"Supported: {', '.join(SUPPORTED_TYPES.values())}",
)

if uploaded:
    text = extract_text_from_file(uploaded)
    if text and text.strip():
        st.success(f"Extracted **{len(text):,}** characters from `{uploaded.name}`")
        with st.expander("Preview text"):
            st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

        col1, col2 = st.columns(2)
        with col1:
            pause_ms = st.slider("Pause between speakers (ms)", 200, 800, 400, 100)
        with col2:
            output_name = st.text_input("Output filename", "narration.mp3", help="Name for the generated MP3")

        if st.button("Generate audio", type="primary"):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                output_path = tmp.name

            with st.spinner("Generating narration... (this may take a minute)"):
                try:
                    run_narrator_agent(
                        input_text=text,
                        output_path=output_path,
                        pause_between_speakers_ms=pause_ms,
                    )
                except Exception as e:
                    st.error(str(e))
                    raise

            with open(output_path, "rb") as f:
                audio_bytes = f.read()

            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                label="Download MP3",
                data=audio_bytes,
                file_name=output_name,
                mime="audio/mp3",
            )

            # Clean up temp file
            try:
                Path(output_path).unlink(missing_ok=True)
            except OSError:
                pass
    elif text is not None:
        st.warning("The file appears to be empty.")

else:
    st.info("👆 Upload a text, PDF, Markdown, or HTML file to get started.")
