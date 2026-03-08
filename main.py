#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import re
import sys
import wave
from typing import Optional

# Hugging Face GPU support for zero-configuration GPU allocation
try:
    from huggingface_hub import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    # Fallback: define a dummy decorator if huggingface_hub isn't installed
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn

# Ensure UTF-8 encoding for output (especially important on Windows)
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Compatibility patch: newer `phonemizer` releases don't expose
# a `set_data_path` classmethod on `EspeakWrapper` while some
# packages (misaki/kokoro) call it at import time. Add a small
# shim so those imports succeed.
# On Windows, also explicitly set espeak-ng paths using espeakng_loader
try:
    import importlib
    import pathlib
    import sys
    
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
    
    # Windows-specific: Set espeak-ng paths from espeakng_loader
    if sys.platform == "win32":
        try:
            import espeakng_loader
            data_path = espeakng_loader.get_data_path()
            if data_path:
                # Set it on the EspeakWrapper
                Esw.set_data_path(str(data_path))
                os.environ["ESPEAK_DATA_PATH"] = str(data_path)
        except (ImportError, Exception):
            pass

except Exception as e:
    # If anything goes wrong here, fall back to normal import and let
    # kokoro raise the original error so the user can see it.
    print(f"Warning: espeak-ng compatibility patch encountered an issue: {e}", file=sys.stderr)
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
- identify if there is dialogue and who is speaking it, but if unsure, default to "NARRATOR" and "narrator_neutral".
Use your best judgment to split the text into meaningful segments based on changes in speaker or narration.
-include dialogue attribution (e.g. "Alice said") when possible to help identify speakers.
- If the text is purely narration with no dialogue, return a single segment with speaker "N
ARRATOR" and speaker_profile "narrator_neutral".
-include emottional tone in speaker_profile when identifiable from the text (e.g. "male_gruff" for a rough, angry male character, "female_warm" for a kind, gentle female character, etc.)
- Focus very very well in the gender and the age of the characters so the voices are consistent and fitting. If you can't identify the gender or age, use "narrator_neutral" for speaker_profile.
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

CHARACTER_METADATA_PROMPT = """
You are an expert literary analyst. Extract and analyze ALL unique characters and narrators from the text.

For EACH character/narrator, return a JSON object with COMPLETE metadata:
- "name": character name or "NARRATOR" for narration
- "type": one of "narrator", "character", or "minor_character"
- "gender": one of "male", "female", "neutral", or "unknown"
- "age_range": one of "child", "teen", "young_adult", "adult", "middle_aged", "elderly", or "unknown"
- "age_estimate": estimated age or age range (e.g., "7-9 years", "30s", "unknown")
- "voice_profile": the speaker_profile from this list:
    narrator_neutral | narrator_male | narrator_female |
    male_young | male_old | male_gruff |
    female_young | female_old | female_warm | child
- "personality_traits": list of personality descriptors (e.g., ["kind", "nervous", "authoritative"])
- "emotional_tone": how they speak (e.g., "warm", "harsh", "cheerful", "sad")
- "role_description": brief description of their role in the text
- "sample_dialogue": one or two lines of their dialogue (or empty string for pure narration)
- "appearance_details": physical description if mentioned (or empty string)
- "relationships": list of other characters they interact with

Return ONLY a valid JSON array of character objects. No markdown, no explanation.

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
        print("[WARNING] No GEMINI_API_KEY/GOOGLE_API_KEY found. Using basic rule-based parser.")
        print("   For best results, set your Gemini API key.\n")
        return _basic_parser(text)


def extract_character_metadata(text: str, api_key: Optional[str] = None) -> list[dict]:
    """
    Extract detailed metadata about all characters and narrators in the text.
    Returns a list of character dictionaries with comprehensive metadata.
    """
    gemini_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=CHARACTER_METADATA_PROMPT.format(text=text[:12000]),
        )
        raw = (getattr(response, "text", None) or str(response)).strip()
        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    else:
        print("[WARNING] No GEMINI_API_KEY/GOOGLE_API_KEY found. Using basic character extraction.")
        print("   For best results, set your Gemini API key.\n")
        return _basic_character_extractor(text)


def _basic_character_extractor(text: str) -> list[dict]:
    """
    Fallback: Extract character metadata using basic regex patterns.
    Returns limited metadata when no API key is available.
    """
    characters = {}
    
    # Find all quoted dialogue to extract speaker names
    dialogue_pattern = r'(\w+(?:\s+\w+)?)\s+(?:said|asked|replied|shouted|whispered|cried|exclaimed|muttered|called|responded)\s*[,:]?\s*"([^"]*)"'
    
    for match in re.finditer(dialogue_pattern, text, re.IGNORECASE):
        speaker_name = match.group(1).title()
        dialogue = match.group(2)
        
        if speaker_name not in characters:
            characters[speaker_name] = {
                "name": speaker_name,
                "type": "character",
                "gender": "unknown",
                "age_range": "unknown",
                "age_estimate": "unknown",
                "voice_profile": "narrator_neutral",
                "personality_traits": [],
                "emotional_tone": "neutral",
                "role_description": f"Character in text",
                "sample_dialogue": dialogue[:100],
                "appearance_details": "",
                "relationships": [],
            }
        else:
            # Update sample dialogue if this one is better
            if len(dialogue) > len(characters[speaker_name].get("sample_dialogue", "")):
                characters[speaker_name]["sample_dialogue"] = dialogue[:100]
    
    # Add narrator if any narration exists
    if not any(c["name"] == "NARRATOR" for c in characters.values()):
        characters["NARRATOR"] = {
            "name": "NARRATOR",
            "type": "narrator",
            "gender": "neutral",
            "age_range": "unknown",
            "age_estimate": "unknown",
            "voice_profile": "narrator_neutral",
            "personality_traits": [],
            "emotional_tone": "neutral",
            "role_description": "Story narrator/description",
            "sample_dialogue": "",
            "appearance_details": "",
            "relationships": [],
        }
    
    return list(characters.values())


def _basic_parser(text: str) -> list[dict]:
    segments = []
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    speaker_counter = {}
    speaker_profiles = {}  # Track consistent profiles for speakers

    for para in paragraphs:
        # Split paragraph into narration and dialogue while preserving order
        parts = []
        last_end = 0
        
        # Find all quoted sections with their positions
        for match in re.finditer(r'"([^"]*)"', para):
            quote_text = match.group(1)
            quote_start = match.start()
            quote_end = match.end()
            
            # Add narration before the quote
            if quote_start > last_end:
                narration = para[last_end:quote_start].strip().strip(',').strip()
                if narration:
                    parts.append(("narration", narration))
            
            # Add the dialogue
            parts.append(("dialogue", quote_text, quote_start, quote_end))
            last_end = quote_end
        
        # Add any remaining narration after the last quote
        if last_end < len(para):
            remaining = para[last_end:].strip().strip(',').strip()
            if remaining:
                parts.append(("narration", remaining))
        
        # If no quotes found, treat entire paragraph as narration
        if not parts:
            segments.append({"speaker": "NARRATOR", "text": para,
                              "speaker_profile": "narrator_neutral", "rate": "normal", "pitch": "normal"})
            continue
        
        # Process parts in order
        for part in parts:
            if part[0] == "narration":
                segments.append({"speaker": "NARRATOR", "text": part[1],
                                  "speaker_profile": "narrator_neutral", "rate": "normal", "pitch": "normal"})
            else:  # dialogue
                quote_text = part[1]
                quote_start = part[2]
                quote_end = part[3]
                
                # Extract speaker from attribution (look backward and forward from quote)
                before_text = para[:quote_start]
                after_text = para[quote_end:]
                
                speaker = None
                
                # Try to find speaker in text before quote
                speaker_match = re.search(
                    r'(\w+)\s+(?:said|asked|replied|shouted|whispered|cried)(?:\s+to\s+\w+)?(?:\s*[:,]?\s*)?$',
                    before_text, re.IGNORECASE
                )
                if speaker_match:
                    speaker = speaker_match.group(1).capitalize()
                else:
                    # Try to find speaker in text after quote
                    speaker_match = re.search(
                        r'^(?:\s*[:,]?\s*)(\w+)\s+(?:said|asked|replied|shouted|whispered|cried)',
                        after_text, re.IGNORECASE
                    )
                    if speaker_match:
                        speaker = speaker_match.group(1).capitalize()
                
                if not speaker:
                    speaker = f"Character_{len(speaker_counter) + 1}"
                
                # Assign profile if not already assigned to this speaker
                if speaker not in speaker_profiles:
                    if speaker not in speaker_counter:
                        speaker_counter[speaker] = len(speaker_counter)
                    profile_options = ["male_young", "female_young", "male_old", "female_warm", "male_gruff"]
                    speaker_profiles[speaker] = profile_options[speaker_counter[speaker] % len(profile_options)]
                
                segments.append({"speaker": speaker, "text": quote_text,
                                  "speaker_profile": speaker_profiles[speaker], "rate": "normal", "pitch": "normal"})
    
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

@spaces.GPU
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


@spaces.GPU
def generate_audio_segments(script: list[dict]) -> list[Optional[np.ndarray]]:
    """
    Generate audio for all segments. Returns list of numpy arrays (or None for empty segments).
    Runs synchronously — Kokoro is local CPU/GPU inference, no async needed.
    GPU-accelerated on Hugging Face Spaces.
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
            print(f"    [WARNING] Segment {i+1} failed: {e}")
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
    Preserves the original order of segments, replacing failed ones with brief silence.
    """
    if not audio_segments:
        raise RuntimeError("No audio segments to stitch.")

    silence = _make_silence(pause_ms)
    pieces  = []
    for i, audio in enumerate(audio_segments):
        # Use the audio if available, otherwise use brief silence to maintain ordering
        if audio is not None and len(audio) > 0:
            pieces.append(audio)
        else:
            pieces.append(_make_silence(100))  # brief 100ms silence for failed segments
        
        # Add pause between segments (but not after the last one)
        if i < len(audio_segments) - 1:
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
    script: Optional[list[dict]] = None,
) -> str:
    """
    Full pipeline: text → Gemini analysis → Kokoro TTS → single WAV file.

    If a pre-computed `script` is provided, the analysis step will be
    skipped and the given script will be used directly. This allows the
    caller (e.g. a UI) to display or inspect the segments beforehand.

    100% local inference — no network calls for TTS, no ffmpeg, no pydub.
    Requires:  pip install kokoro numpy
    Optional:  pip install google-genai python-dotenv  (for Gemini analysis)
    """
    print("=" * 55)
    print("  AI NARRATOR AGENT  (Kokoro TTS)")
    print("=" * 55)

    # ── 1. Analyze ──────────────────────────────────────────
    if script is None:
        print("\n[ANALYSIS] Analyzing text and identifying speakers...")
        script = analyze_text(input_text, api_key=gemini_api_key)
        print(f"   Found {len(script)} segments across {len({s['speaker'] for s in script})} speakers.\n")
    else:
        print("\n[ANALYSIS] Using precomputed script (analysis skipped)")


    print("[SCRIPT] Script preview:")
    for seg in script[:8]:
        preview = seg['text'][:60].replace('\n', ' ')
        print(f"   [{seg['speaker']:<15}] {preview}...")
    if len(script) > 8:
        print(f"   ... and {len(script) - 8} more segments.\n")

    # ── 2. Generate ─────────────────────────────────────────
    print("\n[AUDIO] Generating audio segments (local Kokoro inference)...")
    audio_segments = generate_audio_segments(script)

    # ── 3. Stitch & export ───────────────────────────────────
    print("\n🎧 Stitching into final WAV...")
    final_path = stitch_and_export(audio_segments, output_path, pause_between_speakers_ms)

    print("\n🎉 Done! Your narration is ready:")
    print(f"   → {os.path.abspath(final_path)}\n")
    return final_path


def extract_and_display_characters(
    input_text: str,
    gemini_api_key: Optional[str] = None,
) -> list[dict]:
    """
    Extract and display all characters and their metadata from the text.
    Returns structured character data without generating audio.
    
    Character metadata includes:
    - Name and type (narrator, character, minor_character)
    - Gender and age range with estimate
    - Voice profile assignment
    - Personality traits and emotional tone
    - Role description and appearance details
    - Sample dialogue and relationships
    """
    print("=" * 60)
    print("  CHARACTER METADATA EXTRACTION")
    print("=" * 60)
    
    print("\n[EXTRACTION] Analyzing text for characters and metadata...")
    characters = extract_character_metadata(input_text, api_key=gemini_api_key)
    
    print(f"\n✅ Found {len(characters)} unique character(s):\n")
    
    for i, char in enumerate(characters, 1):
        print(f"{'─' * 60}")
        print(f"[{i}] NAME: {char.get('name', 'UNKNOWN')}")
        print(f"    TYPE: {char.get('type', 'unknown').upper()}")
        print(f"    GENDER: {char.get('gender', 'unknown')}")
        print(f"    AGE: {char.get('age_range', 'unknown')} ({char.get('age_estimate', 'unknown')})")
        print(f"    VOICE: {char.get('voice_profile', 'narrator_neutral')}")
        print(f"    TONE: {char.get('emotional_tone', 'neutral')}")
        print(f"    TRAITS: {', '.join(char.get('personality_traits', [])) or 'none identified'}")
        print(f"    ROLE: {char.get('role_description', 'not specified')}")
        if char.get('appearance_details'):
            print(f"    APPEARANCE: {char.get('appearance_details')}")
        if char.get('sample_dialogue'):
            print(f"    SAMPLE: \"{char.get('sample_dialogue')}\"")
        if char.get('relationships'):
            print(f"    INTERACTIONS: {', '.join(char.get('relationships', []))}")
    
    print(f"\n{'─' * 60}")
    print("\n📊 Character Summary (JSON format):\n")
    print(json.dumps(characters, indent=2))
    
    return characters


# ─────────────────────────────────────────────
#  EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    sample_text = """"This is a simple testing passage to exercise the narrator agent.

    It includes narration and a little dialogue.
    
    The attic was thick with the scent of aged cedar and dust. Sunlight cut through the gloom in a single, sharp beam.
    
    Bob said, "It's unsettling to think that while we’re in this old dust, machines are learning to think and mimic our souls better than we can ourselves."
    
    Alice replied, "It is scary because if computers start doing everything and dreaming for us, will there be any room left for real people to just be quiet and explore places like this?"

    """

    # ─── Option 1: Extract character metadata only ───
    print("\n" + "="*60)
    print("DEMO: Character Extraction")
    print("="*60)
    # Uncomment to extract characters without generating audio:
    # characters = extract_and_display_characters(sample_text)
    
    # ─── Option 2: Run full narrator agent (analyze + TTS) ───
    print("\n" + "="*60)
    print("DEMO: Full Narrator Agent")
    print("="*60)
    run_narrator_agent(
        input_text=sample_text,
        output_path="my_story.wav",
        # gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )
