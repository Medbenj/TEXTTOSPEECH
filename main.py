import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from google import genai
import edge_tts


# ─────────────────────────────────────────────
#  VOICE LIBRARY  (Edge TTS — free, no API key)
# ─────────────────────────────────────────────
VOICE_PROFILES = {
    "narrator_neutral": "en-US-AriaNeural",
    "narrator_male":    "en-US-GuyNeural",
    "narrator_female":  "en-US-JennyNeural",
    "male_young":       "en-US-ChristopherNeural",
    "male_old":         "en-US-EricNeural",
    "male_gruff":       "en-GB-RyanNeural",
    "female_young":     "en-US-SaraNeural",
    "female_old":       "en-US-MichelleNeural",
    "female_warm":      "en-US-CoraNeural",
    "child":            "en-US-AnaNeural",
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
- "pitch": pitch adjustment: "low" | "normal" | "high"

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
            speaker = (speaker_match.group(1) or speaker_match.group(2)).capitalize() if speaker_match else f"Character_{len(speaker_counter) + 1}"

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
#  STEP 2: Rate / pitch maps
# ─────────────────────────────────────────────

RATE_MAP  = {"slow": "-20%", "normal": "+0%", "fast": "+25%"}
PITCH_MAP = {"low":  "-8Hz", "normal": "+0Hz", "high": "+8Hz"}


# ─────────────────────────────────────────────
#  STEP 3: Generate audio segments
# ─────────────────────────────────────────────

def _tts_segment_sync(text: str, voice: str, rate: str, pitch: str, output_path: str):
    import edge_tts.exceptions as edge_exc

    def _write(comm):
        with open(output_path, "wb") as f:
            for chunk in comm.stream_sync():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
    try:
        _write(edge_tts.Communicate(text=text, voice=voice, rate=RATE_MAP.get(rate, "+0%"), pitch=PITCH_MAP.get(pitch, "+0Hz")))
    except edge_exc.NoAudioReceived:
        _write(edge_tts.Communicate(text=text, voice=voice))


TTS_CONCURRENCY = 3
TTS_TIMEOUT_SEC = 90


async def _tts_segment(text, voice, rate, pitch, output_path, semaphore):
    async with semaphore:
        await asyncio.wait_for(
            asyncio.to_thread(_tts_segment_sync, text, voice, rate, pitch, output_path),
            timeout=TTS_TIMEOUT_SEC,
        )


async def generate_audio_segments(script: list[dict], tmp_dir: str) -> list[Optional[str]]:
    semaphore = asyncio.Semaphore(TTS_CONCURRENCY)
    tasks = []
    paths: list[Optional[str]] = []

    for i, seg in enumerate(script):
        voice = VOICE_PROFILES.get(seg.get("speaker_profile", "narrator_neutral"), VOICE_PROFILES["narrator_neutral"])
        text  = seg.get("text", "").strip()
        if text:
            out = os.path.join(tmp_dir, f"seg_{i:04d}.mp3")
            paths.append(out)
            tasks.append(_tts_segment(text, voice, seg.get("rate", "normal"), seg.get("pitch", "normal"), out, semaphore))
            print(f"  [{i+1}/{len(script)}] {seg['speaker']:<15} -> {voice}")
        else:
            paths.append(None)

    await asyncio.gather(*tasks)

    for p in paths:
        if p and (not os.path.exists(p) or os.path.getsize(p) == 0):
            print(f"⚠️  Warning: {p} is empty or missing.")

    return paths


# ─────────────────────────────────────────────
#  STEP 4: Stitch audio
#
#  Strategy waterfall — fastest/safest first:
#  1. Raw MP3 byte concat  (pure Python, no tools, always works)
#  2. FFmpeg subprocess    (if installed, adds silence gaps)
#  pydub is intentionally skipped — it hangs on many Windows setups
# ─────────────────────────────────────────────

def _find_ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return None


def _stitch_with_ffmpeg(valid_paths: list[str], output_path: str, pause_ms: int, ffmpeg_bin: str) -> bool:
    """
    Use ffmpeg subprocess directly with a concat demuxer file.
    Returns True on success, False on any failure.
    Uses a 30-second timeout to prevent hanging.
    """
    try:
        tmp_dir = tempfile.mkdtemp()
        list_file = os.path.join(tmp_dir, "segments.txt")

        # Build the concat list, inserting a silent MP3 between each segment
        # Generate silence segment once
        silence_path = os.path.join(tmp_dir, "silence.mp3")
        silence_cmd = [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
            "-t", str(pause_ms / 1000),
            "-q:a", "9", "-acodec", "libmp3lame",
            silence_path
        ]
        result = subprocess.run(silence_cmd, capture_output=True, timeout=15)
        has_silence = result.returncode == 0 and os.path.exists(silence_path)

        with open(list_file, "w") as f:
            for i, p in enumerate(valid_paths):
                f.write(f"file '{p}'\n")
                if has_silence and i < len(valid_paths) - 1:
                    f.write(f"file '{silence_path}'\n")

        concat_cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]
        result = subprocess.run(concat_cmd, capture_output=True, timeout=30)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            print(f"⚠  FFmpeg concat failed: {result.stderr.decode()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠  FFmpeg timed out. Falling back to raw concat.")
        return False
    except Exception as e:
        print(f"⚠  FFmpeg error: {e}")
        return False


def stitch_audio(segment_paths: list[Optional[str]], output_path: str, pause_ms: int = 400) -> str:
    valid = [p for p in segment_paths if p and os.path.exists(p) and os.path.getsize(p) > 0]
    if not valid:
        raise RuntimeError("No valid audio segments to stitch.")

    print(f"   Stitching {len(valid)} segments...")

    # ── Strategy 1: FFmpeg via subprocess (with timeout, no pydub) ──
    ffmpeg_bin = _find_ffmpeg()
    if ffmpeg_bin:
        print(f"   FFmpeg found at: {ffmpeg_bin}")
        if _stitch_with_ffmpeg(valid, output_path, pause_ms, ffmpeg_bin):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"\n✅ Exported MP3 (ffmpeg): {output_path}  ({size_kb:.0f} KB)")
            return output_path
        print("   Falling back to raw MP3 concat...")

    # ── Strategy 2: Raw MP3 byte concat (always works, no tools) ────
    print("   Using raw MP3 concatenation (pure Python).")
    with open(output_path, "wb") as out_f:
        for p in valid:
            with open(p, "rb") as f:
                out_f.write(f.read())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✅ Exported MP3 (raw concat): {output_path}  ({size_kb:.0f} KB)")
    return output_path


# ─────────────────────────────────────────────
#  MAIN AGENT
# ─────────────────────────────────────────────

def run_narrator_agent(
    input_text: str,
    output_path: str = "output_narration.mp3",
    gemini_api_key: Optional[str] = None,
    pause_between_speakers_ms: int = 400,
) -> str:
    print("=" * 55)
    print("  AI NARRATOR AGENT")
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
    print("\n🎙  Generating audio segments...")
    tmp_dir = tempfile.mkdtemp()
    try:
        segment_paths = asyncio.run(generate_audio_segments(script, tmp_dir))

        # ── 3. Stitch ────────────────────────────────────────
        print("\n🎧 Stitching into final MP3...")
        final_path = stitch_audio(segment_paths, output_path, pause_between_speakers_ms)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

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
        output_path="my_story.mp3",
        # gemini_api_key=os.getenv("GEMINI_API_KEY"),
    )