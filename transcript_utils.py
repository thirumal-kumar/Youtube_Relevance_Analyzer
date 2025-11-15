import subprocess
import json
import re
import os
import yt_dlp

# ============================================
# INTERNAL CLEANING HELPERS
# ============================================

def _clean_text(t: str) -> str:
    """Remove timestamps, HTML tags, newlines, multiple spaces."""
    if not t:
        return ""

    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\d{1,2}:\d{2}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _parse_json3_subs(data):
    """Parse YouTube's json3 subtitle events."""
    cleaned = []
    for event in data.get("events", []):
        if "segs" in event:
            for seg in event["segs"]:
                text = seg.get("utf8", "").strip()
                if text:
                    cleaned.append(text)
    return " ".join(cleaned)


# ============================================
# FAST MODE (yt-dlp)
# ============================================

def fetch_transcript_fast(url: str) -> str:
    """
    Extract subtitles via yt-dlp (fast mode).
    Requires Node.js installed (already verified).
    """

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "json3/srv3/srv2/vtt/srt",
        "outtmpl": "tmp_subs",
        # FIX SABR, web_safari issues, missing URL issues
        "extractor_args": {"youtube": {"player_client": ["default"]}},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)

            subs = result.get("subtitles") or result.get("automatic_captions") or {}
            if not subs:
                return None

            if "en" not in subs:
                return None

            source = subs["en"][0]
            ext = source.get("ext")
            download_url = source.get("url")

            if not download_url:
                return None

            tmp_path = f"tmp_subs.{ext}"

            # Windows-friendly download using curl
            subprocess.run(["curl", "-L", "-s", download_url, "-o", tmp_path], check=False)

            if not os.path.exists(tmp_path):
                return None

            # Parse depending on subtitle extension
            if ext == "json3":
                try:
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    text = _parse_json3_subs(data)
                except:
                    return None
            else:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                text = _clean_text(raw)

            os.remove(tmp_path)
            return text

    except Exception as e:
        print("FAST MODE ERROR:", str(e))
        return None


# ============================================
# WHISPER FALLBACK MODE
# ============================================

def fetch_transcript_whisper(url: str, whisper_model="base") -> str:
    """
    Download audio → transcribe using local whisper.
    """
    audio_file = "tmp_audio.m4a"
    if os.path.exists(audio_file):
        os.remove(audio_file)

    try:
        ydl_opts = {
            "quiet": True,
            "format": "bestaudio/best",
            "outtmpl": audio_file,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except:
        return None

    try:
        import whisper
        model = whisper.load_model(whisper_model)
        result = model.transcribe(audio_file)
        text = result.get("text", "")
        os.remove(audio_file)
        return _clean_text(text)

    except Exception as e:
        print("WHISPER ERROR:", e)
        return None


# ============================================
# MASTER WRAPPER (used by Streamlit App)
# ============================================

def get_youtube_transcript(url: str, mode="fast") -> str:
    """
    Unified transcript fetcher.
    """
    if mode == "fast":
        return fetch_transcript_fast(url)

    if mode == "deep":
        return fetch_transcript_whisper(url)

    return None


# ============================================
# FIXED WRAPPER FOR STREAMLIT (required)
# ============================================

def get_transcript(url: str, mode="fast"):
    """
    Streamlit expects: (text, error_message)
    """
    text = get_youtube_transcript(url, mode)

    if not text:
        return "", "Unable to fetch transcript. Try Deep Mode."

    return text, None
