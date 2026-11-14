"""Addressee voting: "...what do you think, Craig?" -> the next speaker is probably Craig.

Most meetings never repeat introductions, but people call each other by name all the
time. Each time a name is used to address someone, whoever speaks next gets a vote;
talking about what someone "just" said gives the previous speaker half a vote. A name
is assigned only with enough votes and a clear majority (precision over recall: a
wrong name is worse than "Person_2").

votes(lines) -> ({name: {label: weight}}, {name: [evidence lines]})
"""
import re
from collections import defaultdict

NAME = r"([A-Z][a-z]{2,15})"
# A name used to address someone: at the end of a question, after a greeting/thanks,
# alone, or handing over the floor.
_VOCATIVE = [
    re.compile(rf"\b(?:thanks|thank you|cheers|okay|ok|right|so|well|and|hi|hey|yes|yeah|sorry|over to|start with|go ahead|what about|how about)\s*,?\s+{NAME}\s*[?.!,]*\s*$"),
    re.compile(rf"\b(?:think|reckon|say|you|recap on that|anything else)\s*,?\s+{NAME}\s*[?.!]*\s*$"),
    re.compile(rf"^{NAME}\s*[?.!]\s*$"),
    re.compile(rf"\b(?:over to|start with|go ahead|what about|how about)\s+{NAME}\b"),
    re.compile(rf"(?:^|[.?!]\s+){NAME},\s+(?:do|did|can|could|would|what|how|have|are|is|you)\b"),
]
# Talking about what someone just presented: "what Craig's just gone over" -> the previous speaker.
_THIRD = re.compile(rf"\b{NAME}(?:'s)?\s+(?:just|was just|has just)\b")
_SENT = re.compile(r"(?<=[.!?])\s+")
# Said alone as a sentence, these are reactions, not names.
INTERJECTIONS = set("""Great Alright Okay Cool Nice Lovely Excellent Perfect Brilliant Fine Sure Right Yes Yeah Yep Nope
Well Wow Hmm Exactly Absolutely Definitely Indeed Thanks Cheers Sorry Oops Awesome Fantastic Good Super Amazing Interesting
True Totally Fair Done Agreed Correct Wonderful Hello Bye Goodbye Anyway Basically Actually Maybe Probably""".split())


def person_like(lines: list[tuple[str, str, str]], not_names: set[str]) -> set[str]:
    """Capitalised words that look like first names: never written in lower case anywhere in the
    meeting, and capitalised at least once in the middle of a sentence (so not just 'Great.')."""
    lower, mid = set(), set()
    for _, _, text in lines:
        for sent in _SENT.split(text):
            toks = re.findall(r"[A-Za-z']+", sent)
            for k, tok in enumerate(toks):
                tok = tok.removesuffix("'s")
                if tok.islower():
                    lower.add(tok)
                elif (k > 0 or len(toks) == 1) and re.fullmatch(r"[A-Z][a-z]{2,15}", tok):
                    mid.add(tok)
    return {n for n in mid if n.lower() not in lower and n not in not_names and n not in INTERJECTIONS}


def votes(lines: list[tuple[str, str, str]], not_names: set[str], window: int = 2):
    """lines = (clock, label, text). Returns ({name: {label: weight}}, {name: [evidence]})."""
    names = person_like(lines, not_names)
    tally: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    proof: dict[str, list[str]] = defaultdict(list)

    def other(i: int, step: int) -> int | None:
        j = i + step
        while 0 <= j < len(lines) and abs(j - i) <= window:
            if lines[j][1] != lines[i][1]:
                return j
            j += step
        return None

    for i, (clock, label, text) in enumerate(lines):
        for sent in _SENT.split(text):
            for pat in _VOCATIVE:
                for m in pat.finditer(sent.strip()):
                    name = m.group(1)
                    if name not in names:
                        continue
                    j = other(i, +1)
                    if j is not None:
                        tally[name][lines[j][1]] += 1.0
                        proof[name].append(f"[{clock}] {label}: \"{sent.strip()[-80:]}\" -> next speaker {lines[j][1]}")
        for m in _THIRD.finditer(text):
            name = m.group(1)
            if name not in names:
                continue
            j = other(i, -1)
            if j is not None:
                tally[name][lines[j][1]] += 0.5
                proof[name].append(f"[{clock}] {label}: \"{m.group(0)}\" -> previous speaker {lines[j][1]}")
    return tally, proof


def assign(tally, taken_labels: set[str], taken_names: set[str], min_votes: float = 1.5, share: float = 0.6):
    """Greedy, strongest first: a name goes to a label if it has enough votes and most of them."""
    cands = []
    for name, by in tally.items():
        if name.lower() in taken_names:
            continue
        total = sum(by.values())
        for label, v in by.items():
            if label not in taken_labels and v >= min_votes and v / total >= share:
                cands.append((v, name, label))
    out = {}
    for v, name, label in sorted(cands, reverse=True):
        if label in out or name in out.values():
            continue
        out[label] = name
    return out
