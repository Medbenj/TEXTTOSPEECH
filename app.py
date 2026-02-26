"""
AI Narrator Web App
Upload text files or paste text directly and generate AI-narrated audio with Kokoro TTS.
"""

import io
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from main import run_narrator_agent

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Narrator - Kokoro TTS",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

def main():
    # Header
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 30px;">
            <h1>🎙️ AI Narrator</h1>
            <p style="font-size: 18px; color: #666;">
                AI-powered text-to-speech with multiple character voices using Kokoro TTS
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "generated_audio" not in st.session_state:
        st.session_state.generated_audio = None
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    # Create two columns
    col1, col2 = st.columns([1, 1])

    # ─── LEFT COLUMN: INPUT ───
    with col1:
        st.subheader("📝 Input Text")
        
        # Tabs for different input methods
        input_tab1, input_tab2 = st.tabs(["📄 Upload File", "✍️ Paste Text"])
        
        with input_tab1:
            uploaded_file = st.file_uploader(
                "Choose a file to narrate",
                type=list(SUPPORTED_TYPES.keys()),
                help=f"Supported formats: {', '.join(SUPPORTED_TYPES.keys())}"
            )
            
            if uploaded_file is not None:
                with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                    extracted = extract_text_from_file(uploaded_file)
                    if extracted:
                        st.session_state.input_text = extracted
                        st.success(f"✅ Extracted {len(extracted)} characters")
        
        with input_tab2:
            st.session_state.input_text = st.text_area(
                "Paste or type your text here",
                value=st.session_state.input_text,
                height=300,
                placeholder="Enter the text you want to narrate...",
            )

        # Text preview and stats
        if st.session_state.input_text:
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Characters", len(st.session_state.input_text))
            with col_stat2:
                st.metric("Words", len(st.session_state.input_text.split()))
            with col_stat3:
                estimated_mins = len(st.session_state.input_text) / 200  # ~200 chars/min
                st.metric("Est. Duration", f"{estimated_mins:.1f} min")

        # Settings
        st.subheader("⚙️ Settings")
        
        pause_ms = st.slider(
            "Pause between speakers (ms)",
            min_value=100,
            max_value=1000,
            value=400,
            step=50,
            help="Add silence between different speakers"
        )

        gemini_key = st.text_input(
            "Gemini API Key (optional)",
            type="password",
            help="For better speaker detection. Leave empty for rule-based parsing."
        )

        # Generate button
        st.markdown("---")
        generate_button = st.button(
            "🚀 Generate Narration",
            use_container_width=True,
            key="generate_btn",
            type="primary" if st.session_state.input_text else "secondary"
        )

        if generate_button:
            if not st.session_state.input_text:
                st.error("❌ Please enter or upload text first!")
            else:
                with st.spinner("🤖 Analyzing text and generating audio..."):
                    try:
                        # Create temp file for the output
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                        ) as tmp_file:
                            output_path = tmp_file.name

                        # Run the narrator agent
                        api_key = gemini_key if gemini_key else None
                        final_path = run_narrator_agent(
                            input_text=st.session_state.input_text,
                            output_path=output_path,
                            gemini_api_key=api_key,
                            pause_between_speakers_ms=pause_ms,
                        )

                        # Read the generated audio
                        with open(final_path, "rb") as f:
                            st.session_state.generated_audio = f.read()

                        st.success("✅ Audio generated successfully!")

                    except Exception as e:
                        st.error(f"❌ Error generating audio: {str(e)}")

    # ─── RIGHT COLUMN: OUTPUT ───
    with col2:
        st.subheader("🎧 Audio Output")
        
        if st.session_state.generated_audio:
            # Display audio player
            st.audio(st.session_state.generated_audio, format="audio/wav")
            
            # File info
            file_size_mb = len(st.session_state.generated_audio) / (1024 * 1024)
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            with col_info2:
                st.metric("Format", "WAV 16-bit PCM @ 24kHz")

            # Download button
            st.download_button(
                label="⬇️ Download WAV File",
                data=st.session_state.generated_audio,
                file_name="narrated_story.wav",
                mime="audio/wav",
                use_container_width=True,
                key="download_btn",
            )

            # Additional export options
            st.markdown("---")
            st.subheader("📤 Export Options")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                if st.button("🔄 Generate Again", use_container_width=True):
                    st.session_state.generated_audio = None
                    st.rerun()
            
            with col_export2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.generated_audio = None
                    st.session_state.input_text = ""
                    st.rerun()

        else:
            # Empty state
            st.info(
                "👈 **No audio generated yet**\n\n"
                "1. Enter or upload text on the left\n"
                "2. Click 'Generate Narration'\n"
                "3. Listen and download your audio file here"
            )

    # ─── INFO SECTION ───
    st.markdown("---")
    
    with st.expander("ℹ️ About AI Narrator"):
        st.markdown(
            """
            **AI Narrator** uses Kokoro TTS to convert text into high-quality speech.
            
            ### Features:
            - 🎯 **Multi-speaker support** - Different voices for narrator and characters
            - 🌍 **Natural speech** - Powered by Kokoro 82M model
            - ⚡ **Local processing** - Fast, private inference without cloud APIs
            - 📁 **Multiple formats** - Upload TXT, PDF, Markdown, HTML, RTF
            - 🎛️ **Customizable** - Adjust speaker pauses and speech speed
            
            ### Supported Voice Profiles:
            - **American**: af_sarah, af_bella, af_sky, am_adam, am_michael
            - **British**: bf_emma, bf_isabella, bm_george, bm_lewis
            - **Special**: af_heart, af_nicole (child)
            """
        )

    with st.expander("🎤 Voice Profiles & Characters"):
        st.markdown(
            """
            | Profile | Gender | Age | Style |
            |---------|--------|-----|-------|
            | af_sarah | Female | Adult | Neutral, Clear |
            | af_bella | Female | Adult | Warm, Friendly |
            | af_sky | Female | Young | Energetic, Light |
            | af_heart | Female | Adult | Warm, Caring |
            | af_nicole | Female | Child | High-pitched, Young |
            | am_adam | Male | Young | Clear, Young Adult |
            | am_michael | Male | Adult | Deep, Authoritative |
            | bm_george | Male | Adult | British, Formal |
            | bm_lewis | Male | Adult | British, Friendly |
            | bf_emma | Female | Adult | British, Gentle |
            | bf_isabella | Female | Adult | British, Warm |
            """
        )

    with st.expander("⚡ Tips & Tricks"):
        st.markdown(
            """
            - **Use character names**: The AI will detect them and assign different voices
            - **Add dialogue tags**: `[NARRATOR]` or character names help with voice assignment
            - **Break up long texts**: Shorter paragraphs generate faster
            - **Commas and pauses**: Natural punctuation helps with pacing
            - **API Key**: Add a Gemini API key for smarter speaker detection
            """
        )


if __name__ == "__main__":
    main()


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
# Done