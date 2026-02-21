import asyncio
import json
import os
import re
import tempfile

from dotenv import load_dotenv

load_dotenv()  # load .env file into environment
from pathlib import Path
from typing import Optional

from google import genai
import edge_tts
from pydub import AudioSegment

# ─────────────────────────────────────────────
#  VOICE LIBRARY  (Edge TTS — free, no API key)
#  Full list: run `edge-tts --list-voices`
# ─────────────────────────────────────────────
VOICE_PROFILES = {
    "narrator_neutral": "en-US-AriaNeural",        # calm, clear narrator
    "narrator_male":    "en-US-GuyNeural",          # male narrator
    "narrator_female":  "en-US-JennyNeural",        # female narrator
    "male_young":       "en-US-ChristopherNeural",  # young adult male
    "male_old":         "en-US-EricNeural",         # older male, deep
    "male_gruff":       "en-GB-RyanNeural",         # British gruff male
    "female_young":     "en-US-SaraNeural",         # young adult female
    "female_old":       "en-US-MichelleNeural",     # mature female
    "female_warm":      "en-US-CoraNeural",         # warm, expressive female
    "child":            "en-US-AnaNeural",          # child voice
}

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
  (use slow for dramatic/emotional moments, fast for excited/urgent speech)
- "pitch": pitch adjustment: "low" | "normal" | "high"
  (low for serious/old characters, high for excited/young characters)

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
    """
    Send text to Gemini (2.5 Flash) to extract a structured script.
    Falls back to a simple rule-based parser if no API key is provided.
    """
    gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=ANALYSIS_PROMPT.format(text=text[:12000]),  # cap at ~12k chars
        )
        raw = (getattr(response, "text", None) or str(response)).strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    else:
        # ── Fallback: simple dialogue parser (no LLM needed) ──────────────
        print("⚠  No GEMINI_API_KEY/GOOGLE_API_KEY found. Using basic rule-based parser.")
        print("   For best results, set your Gemini API key.\n")
        return _basic_parser(text)


def _basic_parser(text: str) -> list[dict]:
    """
    Simple fallback: splits text into narration vs. dialogue.
    Dialogue is detected by quotation marks.
    """
    segments = []
    # Split on paragraph breaks
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    speaker_counter = {}

    for para in paragraphs:
        # Find all quoted dialogue within paragraph
        quotes = re.findall(r'"([^"]+)"', para)
        if quotes:
            # Extract surrounding narration
            non_quote = re.sub(r'"[^"]+"', '', para).strip().strip(',').strip()
            # Try to detect speaker from "said John" / "John said" patterns
            speaker_match = re.search(
                r'(?:said|asked|replied|shouted|whispered|cried)\s+(\w+)|(\w+)\s+(?:said|asked|replied|shouted|whispered|cried)',
                non_quote, re.IGNORECASE
            )
            if speaker_match:
                speaker = (speaker_match.group(1) or speaker_match.group(2)).capitalize()
            else:
                speaker = f"Character_{len(speaker_counter) + 1}"

            if speaker not in speaker_counter:
                speaker_counter[speaker] = len(speaker_counter)

            # Add narration before dialogue
            if non_quote:
                segments.append({
                    "speaker": "NARRATOR",
                    "text": non_quote,
                    "speaker_profile": "narrator_neutral",
                    "rate": "normal",
                    "pitch": "normal"
                })

            # Alternate voices for unnamed characters
            profile_options = ["male_young", "female_young", "male_old", "female_warm", "male_gruff"]
            profile = profile_options[speaker_counter[speaker] % len(profile_options)]

            for quote in quotes:
                segments.append({
                    "speaker": speaker,
                    "text": quote,
                    "speaker_profile": profile,
                    "rate": "normal",
                    "pitch": "normal"
                })
        else:
            # Pure narration
            segments.append({
                "speaker": "NARRATOR",
                "text": para,
                "speaker_profile": "narrator_neutral",
                "rate": "normal",
                "pitch": "normal"
            })

    return segments


# ─────────────────────────────────────────────
#  STEP 2: Map rate/pitch to SSML adjustments
# ─────────────────────────────────────────────

RATE_MAP  = {"slow": "-20%", "normal": "+0%",  "fast": "+25%"}
PITCH_MAP = {"low":  "-8Hz", "normal": "+0Hz", "high": "+8Hz"}


# ─────────────────────────────────────────────
#  STEP 3: Generate audio for each segment
# ─────────────────────────────────────────────

async def _tts_segment(text: str, voice: str, rate: str, pitch: str, output_path: str):
    """Generate a single audio file using edge-tts."""
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=RATE_MAP.get(rate, "+0%"),
        pitch=PITCH_MAP.get(pitch, "+0Hz"),
    )
    await communicate.save(output_path)


async def generate_audio_segments(script: list[dict], tmp_dir: str) -> list[str]:
    """
    Generate all audio segments concurrently and return ordered file paths.
    """
    tasks = []
    paths = []

    for i, seg in enumerate(script):
        voice   = VOICE_PROFILES.get(seg.get("speaker_profile", "narrator_neutral"), VOICE_PROFILES["narrator_neutral"])
        rate    = seg.get("rate",  "normal")
        pitch   = seg.get("pitch", "normal")
        text    = seg.get("text",  "").strip()
        out     = os.path.join(tmp_dir, f"seg_{i:04d}.mp3")
        paths.append(out)

        if text:
            tasks.append(_tts_segment(text, voice, rate, pitch, out))
            print(f"  [{i+1}/{len(script)}] {seg['speaker']:<20} → {voice}")
        else:
            paths[-1] = None  # skip empty segments

    await asyncio.gather(*tasks)
    return paths


# ─────────────────────────────────────────────
#  STEP 4: Stitch segments into one MP3
# ─────────────────────────────────────────────

def stitch_audio(segment_paths: list[str], output_path: str, pause_ms: int = 400):
    """
    Concatenate all audio segments with a small pause between speakers.
    """
    final = AudioSegment.empty()
    silence = AudioSegment.silent(duration=pause_ms)

    valid_paths = [p for p in segment_paths if p and os.path.exists(p)]
    for i, path in enumerate(valid_paths):
        clip = AudioSegment.from_mp3(path)
        final += clip
        if i < len(valid_paths) - 1:
            final += silence  # pause between segments

    final.export(output_path, format="mp3", bitrate="192k")
    print(f"\n✅ Exported: {output_path}  ({len(final)/1000:.1f}s)")


# ─────────────────────────────────────────────
#  MAIN AGENT
# ─────────────────────────────────────────────

def run_narrator_agent(
    input_text: str,
    output_path: str = "output_narration.mp3",
    gemini_api_key: Optional[str] = None,
    pause_between_speakers_ms: int = 400,
):
    """
    Full pipeline:
      text → analysis → voice script → audio segments → single MP3

    Args:
        input_text:               Raw text (novel, article, story, etc.)
        output_path:              Where to save the final MP3
        gemini_api_key:           Optional. Uses GEMINI_API_KEY / GOOGLE_API_KEY env var if not passed.
        pause_between_speakers_ms: Silence gap between segments in milliseconds.
    """

    print("=" * 55)
    print("  AI NARRATOR AGENT")
    print("=" * 55)

    # ── 1. Analyze ──────────────────────────────────────────
    print("\n📖 Analyzing text and identifying speakers...")
    script = analyze_text(input_text, api_key=gemini_api_key)
    print(f"   Found {len(script)} segments across {len({s['speaker'] for s in script})} speakers.\n")

    # Print the extracted script summary
    print("📝 Script preview:")
    for seg in script[:8]:  # show first 8 for preview
        preview = seg['text'][:60].replace('\n', ' ')
        print(f"   [{seg['speaker']:<15}] {preview}...")
    if len(script) > 8:
        print(f"   ... and {len(script) - 8} more segments.\n")

    # ── 2. Generate audio ───────────────────────────────────
    print("\n🎙  Generating audio segments with Edge TTS...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        segment_paths = asyncio.run(generate_audio_segments(script, tmp_dir))

        # ── 3. Stitch ───────────────────────────────────────
        print("\n🎧 Stitching into final MP3...")
        stitch_audio(segment_paths, output_path, pause_between_speakers_ms)

    print("\n🎉 Done! Your narration is ready:")
    print(f"   → {os.path.abspath(output_path)}\n")
    return output_path


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

    # ── Option A: With Gemini (best quality analysis) ──────
    # Make sure GEMINI_API_KEY (or GOOGLE_API_KEY) is set in your environment, or pass it directly:
    # run_narrator_agent(sample_text, output_path="my_story.mp3", gemini_api_key="YOUR_API_KEY")

    # ── Option B: No API key (uses basic parser) ───────────
    run_narrator_agent(
        input_text=sample_text,
        output_path="my_story.mp3",
        # gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )