"""The steps of the pipeline that need no model: alignment, names, merging, chunking.

They follow the original Colab notebook (AI_Meeting_Assistant.ipynb, cells 7-11).
Where this file differs, the comment on that function says what and why.
"""

import re

Segment = dict  # {"start": float, "end": float, "text": str, "speaker"?: str}


def drop_repeats(segments: list[Segment], keep: int = 2) -> list[Segment]:
    """Whisper sometimes gets stuck and repeats one line many times.

    Not in the notebook. The AMI evaluation found "The last few years." 24 times in a
    row in ES2002a. Keep at most `keep` consecutive copies of the same text.
    """
    out: list[Segment] = []
    run = 0
    for seg in segments:
        same = bool(out) and seg["text"].lower() == out[-1]["text"].lower()
        run = run + 1 if same else 1
        if run <= keep:
            out.append(seg)
    return out


def assign_speakers(segments: list[Segment], turns: list[dict]) -> list[Segment]:
    """Give each transcript segment the speaker who talks most during it (notebook cell 7).

    Overlap is summed per speaker, so a speaker with two short turns inside one segment
    can beat a single slightly longer turn.
    """
    out = []
    for seg in segments:
        overlap: dict[str, float] = {}
        for turn in turns:
            o = min(seg["end"], turn["end"]) - max(seg["start"], turn["start"])
            if o > 0:
                overlap[turn["speaker"]] = overlap.get(turn["speaker"], 0.0) + o
        speaker = max(overlap, key=overlap.get) if overlap else "UNKNOWN"
        out.append({**seg, "speaker": speaker})
    return out


# The notebook searched with re.IGNORECASE, which also made the name's capital letter
# optional, so "I am going to..." named a speaker "Going". Here only the lead-in ignores
# case; the name itself must be capitalised, and common words are refused.
_INTRO = re.compile(r"\b(?i:my name is|my name's|i'm|i am|this is)\s+([A-Z][a-z]{2,15})\b")
_NOT_NAMES = {
    "And", "But", "Going", "Just", "Not", "Really", "Sure", "Sorry", "The", "That", "This",
    "Here", "Fine", "Good", "Okay", "Actually", "Also", "Still", "Very", "Gonna", "Trying",
    "Thinking", "Looking", "Working", "Done", "Back", "Happy", "Glad", "One", "Our", "Your",
}


def name_from_intro(text: str) -> str | None:
    for match in _INTRO.finditer(text):
        name = match.group(1)
        if name not in _NOT_NAMES:
            return name
    return None


def name_speakers(segments: list[Segment]) -> dict[str, str]:
    """Map raw labels (SPEAKER_00) to a name heard in an introduction, else Person_N (cell 8)."""
    found: dict[str, str] = {}
    used: set[str] = set()
    for seg in segments:
        spk = seg["speaker"]
        if spk in found:
            continue
        name = name_from_intro(seg["text"])
        if name and name not in used:
            found[spk] = name
            used.add(name)
    names: dict[str, str] = {}
    n = 1
    # Number the unnamed speakers in the order they first speak, not by label.
    for seg in segments:
        spk = seg["speaker"]
        if spk in names:
            continue
        if spk in found:
            names[spk] = found[spk]
        else:
            names[spk] = f"Person_{n}"
            n += 1
    return names


def assign_words(words: list[dict], turns: list[dict], snap: float = 1.0) -> list[dict]:
    """Give every *word* a speaker, then cut the transcript where the speaker changes.

    Labelling whole Whisper segments instead means a segment that runs across a speaker change
    ("Okay, thanks. / Sure, so the remote...") goes entirely to one person. Here each
    word takes the turn it overlaps most; a word in a gap takes the nearest turn within
    `snap` seconds, else the speaker of the word before it.
    """
    turns = sorted(turns, key=lambda t: t["start"])
    out: list[dict] = []
    prev = "UNKNOWN"
    for w in words:
        best, best_overlap = None, 0.0
        for t in turns:
            if t["start"] > w["end"]:
                break
            o = min(w["end"], t["end"]) - max(w["start"], t["start"])
            if o > best_overlap:
                best, best_overlap = t["speaker"], o
        if best is None:
            mid = (w["start"] + w["end"]) / 2
            near = min(turns, key=lambda t: min(abs(t["start"] - mid), abs(t["end"] - mid)), default=None)
            if near and min(abs(near["start"] - mid), abs(near["end"] - mid)) <= snap:
                best = near["speaker"]
            else:
                best = prev
        prev = best
        out.append({**w, "speaker": best})
    return out


def words_to_segments(words: list[dict], pause: float = 1.5) -> list[Segment]:
    """Group consecutive words of one speaker; a long pause also starts a new segment."""
    segs: list[Segment] = []
    for w in words:
        word = w["word"].strip()
        if not word:
            continue
        last = segs[-1] if segs else None
        if last and last["speaker"] == w["speaker"] and w["start"] - last["end"] <= pause:
            last["text"] += " " + word
            last["end"] = w["end"]
        else:
            segs.append({"start": w["start"], "end": w["end"], "text": word, "speaker": w["speaker"]})
    return segs


def merge_turns(segments: list[Segment]) -> list[Segment]:
    """Join consecutive segments by the same speaker into one turn (cell 9)."""
    merged: list[Segment] = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1] = {**merged[-1], "end": seg["end"], "text": merged[-1]["text"] + " " + seg["text"]}
        else:
            merged.append(dict(seg))
    return merged


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def transcript_lines(turns: list[Segment]) -> list[str]:
    return [f"[{clock(t['start'])}] {t['speaker']}: {t['text']}" for t in turns]


def word_chunks(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """Overlapping word windows for the summariser (cell 10: 400 words, 50 overlap)."""
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break  # the notebook also emitted a tail chunk already inside the last window
    return chunks


def passages(lines: list[str], window: int = 5, stride: int = 3) -> list[str]:
    """Overlapping windows of transcript lines for retrieval (cell 11a)."""
    out = []
    for i in range(0, len(lines), stride):
        block = "\n".join(lines[i : i + window])
        if block:
            out.append(block)
        if i + window >= len(lines):
            break
    return out
