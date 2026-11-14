"""Recording in, meeting minutes out.

    decode       any audio -> 16 kHz mono float32 (soundfile, else ffmpeg)
    transcribe   faster-whisper small, int8, voice-activity filter, a timestamp per word
    diarize      pyannote 3.1: who spoke when (2 to 8 speakers)
    align        each word -> its speaker; the transcript is cut where the speaker changes
    identify     LLM proposes each speaker's name and role with a quote; code verifies it
    minutes      LLM writes title, summary, topics, decisions, actions, open questions
    index        LangChain hybrid retriever (BM25 + FAISS) for questions afterwards

Each stage is a plain function, so the evaluation can run them one by one and reuse
results (for example a diarization computed earlier on a GPU).
"""

import io
import shutil
import subprocess
import time
from typing import Callable

import numpy as np

from echo import llm, minutes, models, speakers, text
from echo.rag import MeetingIndex

RATE = 16000


class PipelineError(RuntimeError):
    """A failure worth showing to the visitor as it is."""


def decode(data: bytes) -> np.ndarray:
    """Any recording -> 16 kHz mono float32."""
    try:
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        # mp3, m4a, mp4 and webm need ffmpeg (installed in the Space's image).
        if not shutil.which("ffmpeg"):
            raise PipelineError("can't read this format here. upload wav, flac or ogg") from None
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "pipe:1"],
            input=data, capture_output=True, timeout=300,
        )
        if out.returncode != 0 or not out.stdout:
            raise PipelineError("couldn't decode that file. is it audio?") from None
        return np.frombuffer(out.stdout, dtype=np.float32).copy()
    if sr != RATE:
        import torch
        import torchaudio.functional as F

        audio = F.resample(torch.from_numpy(audio), sr, RATE).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def transcribe(audio: np.ndarray) -> list[dict]:
    """Every word with its start and end time."""
    segments, _ = models.asr().transcribe(
        audio, language="en", word_timestamps=True, vad_filter=True, beam_size=5,
        # Feeding Whisper its own last sentence is what sets off its repeat loops on long audio.
        condition_on_previous_text=False,
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            if w.word.strip():
                words.append({"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip(),
                              "p": round(w.probability, 3)})
    return words


def diarize(audio: np.ndarray) -> list[dict]:
    import torch

    pipe = models.diarizer()
    pipe.embedding_batch_size = 8
    found = pipe({"waveform": torch.from_numpy(audio)[None], "sample_rate": RATE}, min_speakers=2, max_speakers=8)
    return [
        {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": spk}
        for t, _, spk in found.itertracks(yield_label=True)
    ]


def understand(words: list[dict], turns: list[dict], stage: Callable[[str], None] = lambda s: None,
               timings: dict | None = None) -> dict:
    """Words + speaker turns -> named transcript, minutes, and a question index."""
    timings = {} if timings is None else timings

    def timed(name, fn):
        stage(name)
        t = time.perf_counter()
        out = fn()
        timings[name] = round(time.perf_counter() - t, 1)
        return out

    def align():
        segs = text.words_to_segments(text.assign_words(words, turns))
        return text.merge_turns(segs)

    merged = timed("aligning", align)
    if not merged:
        raise PipelineError("no speech was found to transcribe")
    labelled = [(text.clock(t["start"]), t["speaker"], t["text"]) for t in merged]

    people = timed("identifying", lambda: speakers.identify(labelled))
    turns_out = [{"speaker": people[t["speaker"]]["display"], "label": t["speaker"],
                  "start": text.clock(t["start"]), "start_s": round(t["start"], 2), "end_s": round(t["end"], 2),
                  "text": t["text"]} for t in merged]
    lines = [f"[{t['start']}] {t['speaker']}: {t['text']}" for t in turns_out]
    names = sorted({p["display"] for p in people.values()})

    mins = timed("writing minutes", lambda: minutes.write(lines, names))
    index = timed("indexing", lambda: MeetingIndex(lines))

    talk: dict[str, float] = {}
    for t in turns_out:
        talk[t["speaker"]] = talk.get(t["speaker"], 0.0) + t["end_s"] - t["start_s"]
    return {
        "minutes": {**mins, "transcript": turns_out},
        "speakers": [
            {"label": label, "display": p["display"], "name": p["name"], "role": p["role"],
             "evidence": p["evidence"], "verified": p["verified"], "rejected": p["rejected"],
             "talk_seconds": round(talk.get(p["display"], 0.0), 1)}
            for label, p in sorted(people.items(), key=lambda kv: -talk.get(kv[1]["display"], 0.0))
        ],
        "index": index,
    }


def run(data: bytes, max_seconds: float, on_stage: Callable[[str], None] = lambda s: None) -> dict:
    """The whole pipeline, from uploaded bytes. Returns results plus what was measured."""
    timings: dict[str, float] = {}

    def timed(name, fn):
        on_stage(name)
        t = time.perf_counter()
        out = fn()
        timings[name] = round(time.perf_counter() - t, 1)
        return out

    audio = timed("decoding", lambda: decode(data))
    seconds = len(audio) / RATE
    if seconds < 2:
        raise PipelineError("that recording is shorter than 2 seconds")
    trimmed_from = None
    if seconds > max_seconds:
        trimmed_from = round(seconds, 1)
        audio = audio[: int(max_seconds * RATE)]
        seconds = max_seconds

    words = timed("transcribing", lambda: transcribe(audio))
    if not words:
        raise PipelineError("no speech was heard in that recording")
    turns = timed("diarizing", lambda: diarize(audio))
    before = dict(llm.stats)
    result = understand(words, turns, on_stage, timings)
    return {
        **result,
        "audio_seconds": round(seconds, 1),
        "trimmed_from": trimmed_from,
        "word_count": len(words),
        "speaker_turns": len(turns),
        "stage_seconds": timings,
        "llm": {"model": llm.name(), "calls": llm.stats["calls"] - before["calls"],
                "prompt_tokens": llm.stats["prompt_tokens"] - before["prompt_tokens"],
                "output_tokens": llm.stats["output_tokens"] - before["output_tokens"]},
    }
