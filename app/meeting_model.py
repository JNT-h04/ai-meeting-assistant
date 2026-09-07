import os
import re

from dotenv import load_dotenv
from pyannote.audio import Pipeline
from transformers import pipeline

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. Create a .env file "
        "(see .env.example) with a Hugging Face access token that has "
        "accepted the pyannote/speaker-diarization-3.1 model terms."
    )

# ---------------- LOAD MODELS ONCE ----------------
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    return_timestamps=True,
)

diar_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN,
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# ---------------- HELPERS ----------------
def get_speaker(start, end, speaker_segments):
    max_overlap = 0
    best_speaker = "Unknown"

    for seg in speaker_segments:
        overlap = max(0, min(end, seg["end"]) - max(start, seg["start"]))
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = seg["speaker"]

    return best_speaker


def extract_names(text):
    patterns = [
        r"my name is ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"this is ([A-Z][a-z]+ [A-Z][a-z]+)",
    ]

    for p in patterns:
        match = re.search(p, text)
        if match:
            return match.group(1)

    return None


# ---------------- MAIN FUNCTION ----------------
def process_audio(audio_path):
    # -------- TRANSCRIPTION --------
    result = asr(audio_path, generate_kwargs={"language": "en"})

    segments = []
    for seg in result["chunks"]:
        start, end = seg["timestamp"]

        if start is None or end is None:
            continue

        if (end - start) < 0.5:
            continue

        segments.append({
            "start": start,
            "end": end,
            "text": seg["text"].strip(),
        })

    # -------- DIARIZATION --------
    diarization = diar_pipeline(
        audio_path,
        min_speakers=2,
        max_speakers=8,
    )

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # -------- MAP SPEAKERS --------
    for seg in segments:
        seg["speaker"] = get_speaker(seg["start"], seg["end"], speaker_segments)

    # -------- NAME MAPPING --------
    speaker_name_map = {}

    for seg in segments:
        name = extract_names(seg["text"])
        if name:
            speaker_name_map[seg["speaker"]] = name

    for seg in segments:
        seg["speaker"] = speaker_name_map.get(seg["speaker"], seg["speaker"])

    # -------- MERGE --------
    merged = []
    prev = None

    for seg in segments:
        if prev and seg["speaker"] == prev["speaker"]:
            prev["end"] = seg["end"]
            prev["text"] += " " + seg["text"]
        else:
            if prev:
                merged.append(prev)
            prev = seg

    if prev:
        merged.append(prev)

    # -------- TRANSCRIPT --------
    transcript = "\n".join(
        f"[{s['start']:.2f}-{s['end']:.2f}] {s['speaker']}: {s['text']}" for s in merged
    )

    # -------- SUMMARY --------
    full_text = " ".join(s["text"] for s in merged)

    words = full_text.split()
    chunks = [" ".join(words[i:i + 200]) for i in range(0, len(words), 200)]

    summary = ""
    for chunk in chunks:
        summary += summarizer(chunk, max_length=120, min_length=40)[0]["summary_text"] + " "

    return transcript, summary.strip()
