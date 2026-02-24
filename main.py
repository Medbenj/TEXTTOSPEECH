import json
import os
import re
import wave
from typing import Optional

import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Compatibility patch: newer `phonemizer` releases don't expose
# a `set_data_path` classmethod on `EspeakWrapper` while some
# packages (misaki/kokoro) call it at import time. Add a small
# shim so those imports succeed.
try:
    import importlib, pathlib
    espeak_wrapper = importlib.import_module("phonemizer.backend.espeak.wrapper")
    Esw = espeak_wrapper.EspeakWrapper
    if not hasattr(Esw, "set_data_path"):
        def _set_data_path(cls, path):
            cls._FORCED_DATA_PATH = pathlib.Path(path)
        Esw.set_data_path = classmethod(_set_data_path)
        # prefer forced data path when available
        orig_data_path = Esw.data_path.fget if isinstance(getattr(Esw, 'data_path', None), property) else None
        def _data_path(self):
            if getattr(Esw, "_FORCED_DATA_PATH", None):
                return Esw._FORCED_DATA_PATH
            if orig_data_path:
                return orig_data_path(self)
            return getattr(self, '_data_path', None)
        Esw.data_path = property(_data_path)
except Exception:
    # If anything goes wrong here, fall back to normal import and let
    # kokoro raise the original error so the user can see it.
    pass

from google import genai
from kokoro import KPipeline


# ─────────────────────────────────────────────
#  KOKORO VOICE PROFILES
#  Full list: https://huggingface.co/hexgrad/Kokoro-82M
#
#  American Female : af_heart, af_bella, af_nicole, af_sarah, af_sky
#  American Male   : am_adam, am_michael
#  British Female  : bf_emma, bf_isabella
#  British Male    : bm_george, bm_lewis
# ─────────────────────────────────────────────
VOICE_PROFILES = {
    "narrator_neutral": "af_sarah",
    "narrator_male":    "am_michael",
    "narrator_female":  "af_bella",
    "male_young":       "am_adam",
    "male_old":         "am_michael",
    "male_gruff":       "bm_george",
    "female_young":     "af_sky",
    "female_old":       "bf_emma",
    "female_warm":      "af_heart",
    "child":            "af_nicole",
}

SPEED_MAP = {"slow": 0.8, "normal": 1.0, "fast": 1.2}

# Kokoro sample rate is always 24000 Hz
SAMPLE_RATE = 24000


# ─────────────────────────────────────────────
#  STEP 1: Analyze text with Gemini
# ─────────────────────────────────────────────

ANALYSIS_PROMPT = """
You are an expert literary analyst and audio drama director.

Analyze the following text and return a JSON array of segments.

For each segment identify:
- "speaker": the name/label of who is speaking (use "NARRATOR" for narration/description)
- "text": the exact text of that segment
- "speaker_profile": a brief label from this list that best fits the speaker:
    narrator_neutral | narrator_male | narrator_female |
    male_young | male_old | male_gruff |
    female_young | female_old | female_warm | child
- "rate": speaking rate adjustment, one of: "slow" | "normal" | "fast"
- "pitch": pitch adjustment: "low" | "normal" | "high"
  (note: Kokoro does not support pitch shifting — this field is stored but ignored)

Rules:
- Split the text into meaningful segments. Each new speaker = new segment.
- Consecutive narration can be merged into one segment.
- Keep character voices consistent throughout the text.
- Return ONLY valid JSON — no markdown, no explanation, just the JSON array.

TEXT TO ANALYZE:
\"\"\"
{text}
\"\"\"
"""


def analyze_text(text: str, api_key: Optional[str] = None) -> list[dict]:
    gemini_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=ANALYSIS_PROMPT.format(text=text[:12000]),
        )
        raw = (getattr(response, "text", None) or str(response)).strip()
        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    else:
        print("⚠  No GEMINI_API_KEY/GOOGLE_API_KEY found. Using basic rule-based parser.")
        print("   For best results, set your Gemini API key.\n")
        return _basic_parser(text)


def _basic_parser(text: str) -> list[dict]:
    segments = []
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    speaker_counter = {}

    for para in paragraphs:
        quotes = re.findall(r'"([^"]+)"', para)
        if quotes:
            non_quote = re.sub(r'"[^"]+"', '', para).strip().strip(',').strip()
            speaker_match = re.search(
                r'(?:said|asked|replied|shouted|whispered|cried)\s+(\w+)|(\w+)\s+(?:said|asked|replied|shouted|whispered|cried)',
                non_quote, re.IGNORECASE
            )
            speaker = (
                (speaker_match.group(1) or speaker_match.group(2)).capitalize()
                if speaker_match else f"Character_{len(speaker_counter) + 1}"
            )
            if speaker not in speaker_counter:
                speaker_counter[speaker] = len(speaker_counter)
            if non_quote:
                segments.append({"speaker": "NARRATOR", "text": non_quote,
                                  "speaker_profile": "narrator_neutral", "rate": "normal", "pitch": "normal"})
            profile_options = ["male_young", "female_young", "male_old", "female_warm", "male_gruff"]
            profile = profile_options[speaker_counter[speaker] % len(profile_options)]
            for quote in quotes:
                segments.append({"speaker": speaker, "text": quote,
                                  "speaker_profile": profile, "rate": "normal", "pitch": "normal"})
        else:
            segments.append({"speaker": "NARRATOR", "text": para,
                              "speaker_profile": "narrator_neutral", "rate": "normal", "pitch": "normal"})
    return segments


# ─────────────────────────────────────────────
#  STEP 2: Load Kokoro pipeline (cached)
# ─────────────────────────────────────────────

_pipeline_cache: dict[str, KPipeline] = {}

def _get_pipeline(lang_code: str = "a") -> KPipeline:
    """
    lang_code: 'a' = American English, 'b' = British English
    Pipelines are cached so the model loads only once per language.
    """
    if lang_code not in _pipeline_cache:
        print(f"   Loading Kokoro model (lang='{lang_code}')...")
        _pipeline_cache[lang_code] = KPipeline(lang_code=lang_code)
    return _pipeline_cache[lang_code]


def _voice_lang(voice: str) -> str:
    """Infer lang_code from voice prefix: af/am → 'a', bf/bm → 'b'."""
    return "b" if voice.startswith("b") else "a"


# ─────────────────────────────────────────────
#  STEP 3: Generate audio — fully local,
#          returns numpy float32 arrays
# ─────────────────────────────────────────────

def _generate_segment(text: str, voice: str, speed: float) -> np.ndarray:
    """
    Generate audio for one text segment using Kokoro.
    Returns a float32 numpy array at SAMPLE_RATE Hz.
    """
    lang = _voice_lang(voice)
    pipeline = _get_pipeline(lang)

    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None and len(audio) > 0:
            chunks.append(audio)

    if not chunks:
        raise RuntimeError(f"Kokoro returned no audio for: {text[:50]!r}")

    return np.concatenate(chunks)


def generate_audio_segments(script: list[dict]) -> list[Optional[np.ndarray]]:
    """
    Generate audio for all segments. Returns list of numpy arrays (or None for empty segments).
    Runs synchronously — Kokoro is local CPU/GPU inference, no async needed.
    """
    results: list[Optional[np.ndarray]] = []

    for i, seg in enumerate(script):
        voice = VOICE_PROFILES.get(
            seg.get("speaker_profile", "narrator_neutral"),
            VOICE_PROFILES["narrator_neutral"]
        )
        speed = SPEED_MAP.get(seg.get("rate", "normal"), 1.0)
        text  = seg.get("text", "").strip()

        print(f"  [{i+1}/{len(script)}] {seg['speaker']:<15} -> {voice}  (speed={speed})")

        if not text:
            results.append(None)
            continue

        try:
            audio = _generate_segment(text, voice, speed)
            results.append(audio)
        except Exception as e:
            print(f"    ⚠️  Segment {i+1} failed: {e}")
            results.append(None)

    return results


# ─────────────────────────────────────────────
#  STEP 4: Stitch and export
#  Pure numpy + built-in wave module.
#  No ffmpeg. No pydub. No external tools.
# ─────────────────────────────────────────────

def _make_silence(duration_ms: int) -> np.ndarray:
    """Return a numpy array of silence at SAMPLE_RATE."""
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    return np.zeros(n_samples, dtype=np.float32)


def stitch_and_export(
    audio_segments: list[Optional[np.ndarray]],
    output_path: str,
    pause_ms: int = 400,
) -> str:
    """
    Concatenate all audio segments with silence gaps,
    then write a 16-bit PCM WAV using Python's built-in wave module.
    """
    valid = [a for a in audio_segments if a is not None and len(a) > 0]
    if not valid:
        raise RuntimeError("No valid audio segments to stitch.")

    silence = _make_silence(pause_ms)
    pieces  = []
    for i, audio in enumerate(valid):
        pieces.append(audio)
        if i < len(valid) - 1:
            pieces.append(silence)

    combined = np.concatenate(pieces)

    # Clamp to [-1, 1] and convert to 16-bit PCM
    combined = np.clip(combined, -1.0, 1.0)
    pcm = (combined * 32767).astype(np.int16)

    # Determine output format
    if output_path.lower().endswith(".wav"):
        wav_path = output_path
    else:
        # Write WAV first, note it in output (WAV is lossless and universally supported)
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    duration_s = len(combined) / SAMPLE_RATE
    size_kb    = os.path.getsize(wav_path) / 1024
    print(f"\n✅ Exported WAV: {wav_path}  ({duration_s:.1f}s, {size_kb:.0f} KB)")
    return wav_path


# ─────────────────────────────────────────────
#  MAIN AGENT
# ─────────────────────────────────────────────

def run_narrator_agent(
    input_text: str,
    output_path: str = "output_narration.wav",
    gemini_api_key: Optional[str] = None,
    pause_between_speakers_ms: int = 400,
) -> str:
    """
    Full pipeline: text → Gemini analysis → Kokoro TTS → single WAV file.

    100% local inference — no network calls for TTS, no ffmpeg, no pydub.
    Requires:  pip install kokoro numpy
    Optional:  pip install google-genai python-dotenv  (for Gemini analysis)
    """
    print("=" * 55)
    print("  AI NARRATOR AGENT  (Kokoro TTS)")
    print("=" * 55)

    # ── 1. Analyze ──────────────────────────────────────────
    print("\n📖 Analyzing text and identifying speakers...")
    script = analyze_text(input_text, api_key=gemini_api_key)
    print(f"   Found {len(script)} segments across {len({s['speaker'] for s in script})} speakers.\n")

    print("📝 Script preview:")
    for seg in script[:8]:
        preview = seg['text'][:60].replace('\n', ' ')
        print(f"   [{seg['speaker']:<15}] {preview}...")
    if len(script) > 8:
        print(f"   ... and {len(script) - 8} more segments.\n")

    # ── 2. Generate ─────────────────────────────────────────
    print("\n🎙  Generating audio segments (local Kokoro inference)...")
    audio_segments = generate_audio_segments(script)

    # ── 3. Stitch & export ───────────────────────────────────
    print("\n🎧 Stitching into final WAV...")
    final_path = stitch_and_export(audio_segments, output_path, pause_between_speakers_ms)

    print("\n🎉 Done! Your narration is ready:")
    print(f"   → {os.path.abspath(final_path)}\n")
    return final_path


# ─────────────────────────────────────────────
#  EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    sample_text = """
    The old lighthouse stood at the edge of the world, battered by wind and salt.

    Thomas wiped the fog from the window and peered into the darkness.
    "She's coming," he muttered, his voice raw from a night of watching.

    Elena stepped into the doorway, her coat dripping with rain.
    "You've been up all night again, haven't you?" she said quietly.

    Thomas turned. He looked older than she remembered.
    "Someone has to," he replied. "The ships don't stop just because we're tired."

    Elena crossed the room and placed her hand on his shoulder.
    "The coast guard called. They said the storm will pass by morning."

    He let out a long breath.
    "It always does," Thomas said. "But morning feels very far away tonight."

    They stood there together, watching the dark water churn below.
    The light kept turning. The world kept spinning. And the ships, somewhere out there in the deep, found their way home.
    """

    run_narrator_agent(
        input_text=sample_text,
        output_path="my_story.wav",
        # gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )
    #