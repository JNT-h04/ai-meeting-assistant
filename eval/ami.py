"""Reference data from the AMI Meeting Corpus manual annotations (v1.6.2).

Three things are read per meeting, all written by people, not models:
  - words:     every word each participant said, with its time    -> WER
  - segments:  when each participant was talking                  -> DER
  - abstract:  a hand-written summary of the meeting              -> ROUGE
"""

import html
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ANNOTATIONS = Path(r"D:\Files\ami dataset\annotations")
AUDIO = Path(r"D:\Files\ami dataset\amicorpus")

# Held-out test meetings, downloaded from the AMI mirror (D:\Files stays read-only).
TEST_AUDIO = Path(__file__).resolve().parents[1] / "data" / "ami_test"

# dev: the meetings every rule was developed and tuned on.
# test: never looked at while building; the frozen code is run on them once.
DEV = ["ES2002a", "ES2002b", "ES2002c", "ES2003a", "ES2003b", "ES2003c"]
TEST = ["ES2004a", "ES2004c", "ES2005a", "ES2005c"]
SPLIT = os.environ.get("AMI_SET", "dev")
MEETINGS = TEST if SPLIT == "test" else DEV
NITE = "{http://nite.sourceforge.net/}"


def audio_path(meeting: str) -> Path:
    for root in (AUDIO, TEST_AUDIO):
        path = root / meeting / "audio" / f"{meeting}.Mix-Headset.wav"
        if path.exists():
            return path
    return AUDIO / meeting / "audio" / f"{meeting}.Mix-Headset.wav"


def _speakers(meeting: str) -> list[str]:
    """The participant letters (A, B, C, D) that have a words file."""
    return sorted(p.name.split(".")[1] for p in (ANNOTATIONS / "words").glob(f"{meeting}.*.words.xml"))


def words(meeting: str) -> list[tuple[float, str, str]]:
    """(start time, speaker, word) for every spoken word, in time order. Punctuation is dropped."""
    out = []
    for spk in _speakers(meeting):
        root = ET.parse(ANNOTATIONS / "words" / f"{meeting}.{spk}.words.xml").getroot()
        for w in root.iter("w"):
            if w.get("punc") == "true" or not w.text:
                continue
            start = w.get("starttime")
            out.append((float(start) if start else float("nan"), spk, html.unescape(w.text)))
    # a handful of words have no timestamp; keep them where they appear by sorting stably on the rest
    return sorted(out, key=lambda t: (t[0] != t[0], t[0]))


def reference_text(meeting: str) -> str:
    return " ".join(w for _, _, w in words(meeting))


def turns(meeting: str) -> list[tuple[float, float, str]]:
    """(start, end, speaker) for every stretch of talk, from the segments annotation."""
    out = []
    for spk in _speakers(meeting):
        root = ET.parse(ANNOTATIONS / "segments" / f"{meeting}.{spk}.segments.xml").getroot()
        for seg in root.iter("segment"):
            start, end = float(seg.get("transcriber_start")), float(seg.get("transcriber_end"))
            if end > start:
                out.append((start, end, spk))
    return sorted(out)


def abstract(meeting: str) -> str:
    """The hand-written abstract (the 'abstract' block of the abstractive summary)."""
    root = ET.parse(ANNOTATIONS / "abstractive" / f"{meeting}.abssumm.xml").getroot()
    block = root.find("abstract")
    sentences = [s.text.strip() for s in block.iter("sentence") if s.text] if block is not None else []
    return re.sub(r"\s+", " ", " ".join(sentences)).strip()


def abstractive(meeting: str, block: str) -> list[str]:
    """Sentences of one block of the hand-written summary: abstract, decisions, actions or problems."""
    root = ET.parse(ANNOTATIONS / "abstractive" / f"{meeting}.abssumm.xml").getroot()
    found = root.find(block)
    if found is None:
        return []
    return [re.sub(r"\s+", " ", s.text).strip() for s in found.iter("sentence") if s.text and s.text.strip()]


def participants(meeting: str) -> dict[str, tuple[str, str]]:
    """{letter: (role, global participant id)} from the corpus metadata."""
    root = ET.parse(ANNOTATIONS / "corpusResources" / "meetings.xml").getroot()
    for mt in root.iter("meeting"):
        if mt.get("observation") == meeting:
            return {sp.get("nxt_agent"): (sp.get("role"), sp.get("global_name")) for sp in mt}
    return {}


if __name__ == "__main__":
    for m in MEETINGS:
        t = turns(m)
        print(
            f"{m}: {len(words(m)):5d} words, {len(t):4d} turns, "
            f"{len(_speakers(m))} speakers, audio {audio_path(m).exists()}, "
            f"abstract {len(abstract(m).split())} words"
        )
