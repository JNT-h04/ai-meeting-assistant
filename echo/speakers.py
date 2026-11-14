"""Who is SPEAKER_00? Names and roles, proposed by the LLM and checked against the transcript.

A single regex ("my name is X") misses "I'm Laura" and names people "Going". This works in
three steps:

  1. shortlist  keep the lines that can say who someone is (introductions, names,
                job titles, people being addressed) plus each speaker's first lines,
                so an hour-long meeting still fits the model's context
  2. propose    the LLM returns, for every label, a name, a role and an exact quote
  3. verify     code, not the model, decides what to keep:
                  - the quote must really be in the transcript (fuzzy, >= 0.85)
                  - the name must really be in the transcript
                  - a self-introduction ("I'm X") must come from that same speaker
                  - one name can't go to two speakers
                Anything that fails becomes "unknown" rather than a confident guess.
"""

import re
from difflib import SequenceMatcher

from echo import llm
from echo.address import INTERJECTIONS
from echo.address import assign as assign_votes
from echo.address import votes as address_votes
from echo.text import _NOT_NAMES

# Job titles the verifier can read straight from a quote when the model left the role blank.
ROLE_WORDS = [
    "project manager", "product manager", "program manager", "team lead", "tech lead", "manager",
    "industrial designer", "user interface designer", "user interface", "ux designer", "ui designer",
    "interface designer", "designer", "marketing expert", "marketing executive", "marketing",
    "engineer", "developer", "data scientist", "analyst", "researcher", "scrum master", "director",
    "ceo", "cto", "founder", "intern", "consultant", "sales", "secretary", "chair", "facilitator",
]
_ROLE_RE = re.compile(r"\b(" + "|".join(re.escape(r) for r in ROLE_WORDS) + r")\b", re.I)
_CUE = re.compile(
    r"\b(i'm|i am|my name|name's|this is|call me|introduce|thanks?|thank you|cheers|role|"
    + "|".join(re.escape(r) for r in ROLE_WORDS) + r")\b", re.I)
_SELF = re.compile(r"\b(i'm|i am|my name|name's|call me)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")
# First-person statements, read straight from a speaker's own words.
_SELF_ROLE = re.compile(r"\b(?:i'm|i am)\s+(?:supposed to be\s+|going to be\s+|gonna be\s+|kind of\s+)?"
                        r"(?:a|an|the|our|your)?\s*(" + "|".join(re.escape(r) for r in ROLE_WORDS) + r")\b", re.I)
_SELF_NAME = re.compile(r"\b(?i:my name is|my name's|i'm|i am|call me)\s+([A-Z][a-z]{2,15})\b")
# Only the fillers ignore case: with re.I on the whole pattern "probably" and "you" passed as names.
_ROLL_CALL = re.compile(r"^(?:(?i:uh|um|er|oh|yeah|hi|hello|so|okay|right)[,.]?\s+)*([A-Z][a-z]{2,15})\b")
UNKNOWN = "unknown"

SYSTEM = (
    "You identify the people in a meeting transcript. Each line is '[mm:ss] LABEL: words'. "
    "For every label give: name = the person's first name if they say it about themselves or someone "
    "clearly addresses them by it, else \"unknown\"; role = their job or role in the meeting if they "
    "state it or it is clearly implied, else \"unknown\"; evidence = one short exact quote copied from "
    "the transcript that shows the name or role, else \"\". Never guess a name that is not in the transcript."
)


def _excerpt(text: str, limit: int = 400) -> str:
    """A long turn shrunk to its first sentence plus every sentence with a cue in it."""
    if len(text) <= limit:
        return text
    sents = _SENT.split(text)
    keep = [sents[0]] + [x for x in sents[1:] if _CUE.search(x)]
    return " ... ".join(keep)[: limit * 2]


def shortlist(lines: list[tuple[str, str, str]], budget_tokens: int = 3000) -> str:
    """lines = (clock, label, text). Keep cue lines, each speaker's first 3 lines, and neighbours."""
    keep = set()
    seen: dict[str, int] = {}
    for i, (_, label, text) in enumerate(lines):
        seen[label] = seen.get(label, 0) + 1
        if seen[label] <= 3:
            keep.add(i)
        if _CUE.search(text):
            # the line before/after shows who was being addressed or who answered
            keep.update({i - 1, i, i + 1})
    picked = [lines[i] for i in sorted(k for k in keep if 0 <= k < len(lines))]
    out, used = [], 0
    for clock, label, text in picked:
        line = f"[{clock}] {label}: {_excerpt(text)}"
        used += len(line) // 4 + 1  # ~4 characters a token; exact counts would cost a model call
        if used > budget_tokens:
            break
        out.append(line)
    return "\n".join(out)


def _found(quote: str, haystack: list[str]) -> str | None:
    """The transcript line containing `quote` (fuzzy), or None."""
    q = " ".join(quote.lower().split())
    if len(q) < 4:
        return None
    for line in haystack:
        l = line.lower()
        if q in l:
            return line
        m = SequenceMatcher(None, q, l, autojunk=False).find_longest_match(0, len(q), 0, len(l))
        if m.size >= 0.85 * len(q):
            return line
    return None


def _quote(lines: list[str], fragment: str) -> str:
    """The sentence containing `fragment`, from the speaker's own lines."""
    for t in lines:
        for sent in _SENT.split(t):
            if fragment.lower() in sent.lower():
                return sent.strip()
    return fragment


GENERIC_ROLES = {"team member", "member", "participant", "speaker", "attendee", "person", "colleague",
                 "chairman of the meeting", "meeting participant", "presenter", "listener", "guest", "unknown"}
_PREFIX = re.compile(r"^\[?\d{1,2}:\d\d\]?\s*[A-Za-z_]+[ _]?\d*:\s*")


def _first_sentence(text: str) -> str:
    return _SENT.split(text.strip(), 1)[0]


def _roll_call(text: str) -> str | None:
    """The name a turn opens with, when the turn starts as a roll-call answer.

    "Ebenezer at Demisoy. Do you want me to spell it?" -> Ebenezer. The first sentence
    must be short and start with the name. "Craig, what do you think?" is someone being
    addressed, not answering, so a name followed by a comma doesn't count.
    """
    first = _first_sentence(text)
    m = _ROLL_CALL.match(first)
    if not m or len(first.split()) > 8:  # "Tarek Raman, T -A -R -I -K," is 7
        return None
    if first[m.end():m.end() + 1] == ",":
        return None
    return m.group(1)


_INVITE = re.compile(r"\b(?:next we have|lastly we have|and you are|and you're|who are you|your name|introduce|"
                     r"go round|go around|who's next|and you\??)\s*[?.]?\s*$", re.I)


def _corroborated(name: str, label: str, text: str, lines: list[tuple[str, str, str]]) -> bool:
    """A roll-call name found without the model: someone else also says it, or it answers an invitation."""
    has = re.compile(rf"\b{re.escape(name)}\b")
    if any(who != label and has.search(t) for _, who, t in lines):
        return True
    i = next((k for k, (_, who, t) in enumerate(lines) if who == label and t == text), None)
    prev = next((lines[k][2] for k in range((i or 0) - 1, -1, -1) if lines[k][1] != label), "") if i else ""
    return bool(_INVITE.search(prev))


def _role_supported(role: str, label: str, lines: list[tuple[str, str, str]]) -> bool:
    """Every word of the role is said by the speaker, or by whoever speaks right before or after
    them ("And your role in this is Industrial Designer")."""
    words = [w for w in re.findall(r"[a-z]+", role.lower()) if len(w) > 3 and w not in ("meeting", "team")]
    if not words:
        return False
    for i, (_, who, text) in enumerate(lines):
        near = who == label or any(0 <= j < len(lines) and lines[j][1] == label for j in (i - 1, i + 1))
        if near and all(re.search(rf"\b{w}", text.lower()) for w in words):
            return True
    return False


def _name_supported(name: str, label: str, lines: list[tuple[str, str, str]]) -> tuple[bool, str]:
    """Is there evidence in the transcript tying `name` to `label`?

    Yes if the speaker introduces themselves with it, or another speaker says the name in
    the turn just before or after one of theirs (being addressed: "Thanks, Laura").
    No if the only introduction with that name is someone else's.
    """
    has = re.compile(rf"\b{re.escape(name)}\b", re.I)
    intro_by_other = False
    for i, (_, who, text) in enumerate(lines):
        if not has.search(text):
            continue
        selfie = any(m.group(1).lower() == name.lower() for m in _SELF_NAME.finditer(text))
        if who == label and selfie:
            return True, ""
        if who == label and (_roll_call(text) or "").lower() == name.lower():
            return True, ""  # answering a roll call: "Uh, Dave Cochrane."
        if who != label and selfie:
            intro_by_other = True
            continue
        if who != label and any(0 <= j < len(lines) and lines[j][1] == label for j in (i - 1, i + 1)):
            return True, ""
    if intro_by_other:
        return False, "introduction belongs to another speaker"
    return False, f"nothing ties '{name}' to this speaker"


def _role_from(text: str) -> str:
    m = _ROLE_RE.search(text or "")
    return m.group(1).lower() if m else UNKNOWN


def identify(lines: list[tuple[str, str, str]]) -> dict[str, dict]:
    """lines = (clock, label, text) in order. Returns {label: {name, role, evidence, verified, display}}."""
    labels = sorted({label for _, label, _ in lines})
    person = {"type": "object", "properties": {
        "name": {"type": "string"}, "role": {"type": "string"}, "evidence": {"type": "string"}},
        "required": ["name", "role", "evidence"]}
    schema = {"type": "object", "properties": {label: person for label in labels}, "required": labels}
    proposed = llm.json_call(SYSTEM, shortlist(lines), schema, max_tokens=60 * len(labels) + 80)

    by_label: dict[str, list[str]] = {label: [] for label in labels}
    for _, label, text in lines:
        by_label[label].append(text)
    every_line = [t for _, _, t in lines]
    corpus = " ".join(every_line).lower()

    result: dict[str, dict] = {}
    for label in labels:
        p = proposed.get(label) or {}
        name = (p.get("name") or "").strip().split(" ")[0].strip(".,")
        role = (p.get("role") or "").strip().lower()
        evidence = _PREFIX.sub("", (p.get("evidence") or "").strip())  # models copy "[08:57] SPEAKER_01:"
        why = []

        line = _found(evidence, every_line) if evidence else None
        if evidence and line is None:
            why.append("quote not in transcript")
            evidence = ""
        own = line is not None and line in by_label[label]

        if name.lower() in ("", UNKNOWN, "none", "n/a") or not name[:1].isupper():
            name = UNKNOWN
        elif not re.search(rf"\b{re.escape(name.lower())}\b", corpus):
            why.append(f"name '{name}' never said")
            name = UNKNOWN
        else:
            ok, reason = _name_supported(name, label, lines)
            if not ok:
                why.append(reason)
                name = UNKNOWN
        if name == UNKNOWN and not own:
            evidence = ""  # someone else's words can't stand as this speaker's evidence

        if role in GENERIC_ROLES or role in ("", "none", "n/a"):
            role = UNKNOWN
        elif not _role_supported(role, label, lines):
            why.append(f"role '{role}' not stated by or to this speaker")
            role = UNKNOWN
        if role == UNKNOWN and own:
            role = _role_from(line or "")
        if role == UNKNOWN:
            # The model missed it, but the speaker said it: "I'm an industrial designer".
            m = next((m for t in by_label[label] if (m := _SELF_ROLE.search(t))), None)
            if m:
                role, evidence = m.group(1).lower(), evidence or _quote(by_label[label], m.group(0))
                own = own or bool(evidence)
        result[label] = {"name": name, "role": role, "evidence": evidence, "own_quote": own, "rejected": why}

    # The model said unknown, but the speaker introduced themselves: take it, with the quote.
    taken = {r["name"].lower() for r in result.values() if r["name"] != UNKNOWN}
    # Words this meeting also uses in lower case ("you", "sounds") are not names, however they're capitalised.
    everyday = {w for _, _, t in lines for w in re.findall(r"[A-Za-z']+", t) if w.islower()}
    for label, r in result.items():
        if r["name"] != UNKNOWN:
            continue
        for t in by_label[label]:
            m = _SELF_NAME.search(t)
            found = m.group(1) if m else _roll_call(t)
            if found and not m and not _corroborated(found, label, t, lines):
                found = None  # "Sounds good to me." opens like a roll-call answer too
            if found and found.lower() not in taken and found.lower() not in everyday \
                    and found not in _NOT_NAMES and found not in INTERJECTIONS:
                r["name"], r["own_quote"] = found, True
                r["evidence"] = r["evidence"] or (_quote(by_label[label], m.group(0)) if m else _first_sentence(t))
                taken.add(found.lower())
                break

    # Still unnamed: who answers when a name is called? ("...what do you think, Craig?")
    tally, proof = address_votes(lines, _NOT_NAMES)
    unnamed = {l for l, r in result.items() if r["name"] == UNKNOWN}
    for label, name in assign_votes(tally, {l for l in result if l not in unnamed}, taken).items():
        r = result[label]
        r["name"], r["own_quote"] = name, True  # evidence is about this speaker, not someone else's intro
        r["evidence"] = r["evidence"] or proof[name][0]
        r["named_by"] = f"addressed {tally[name][label]:g}x"
        taken.add(name.lower())

    # One name, one person: keep it where the quote is the speaker's own words.
    names: dict[str, list[str]] = {}
    for label, r in result.items():
        if r["name"] != UNKNOWN:
            names.setdefault(r["name"].lower(), []).append(label)
    for owners in names.values():
        if len(owners) > 1:
            winner = next((l for l in owners if result[l]["own_quote"]), None)
            for l in owners:
                if l != winner:
                    result[l]["rejected"].append("name also given to another speaker")
                    result[l]["name"] = UNKNOWN

    # What the transcript shows: name, else role, else Person_N in speaking order.
    order = list(dict.fromkeys(label for _, label, _ in lines))
    n = 0
    for label in order:
        r = result[label]
        r["verified"] = r["name"] != UNKNOWN or r["role"] != UNKNOWN
        if r["name"] != UNKNOWN:
            r["display"] = r["name"]
        elif r["role"] != UNKNOWN:
            r["display"] = r["role"].title()
        else:
            n += 1
            r["display"] = f"Person_{n}"
    # Two speakers with the same role-only display ("Designer") get numbered.
    shown: dict[str, list[str]] = {}
    for label in order:
        shown.setdefault(result[label]["display"], []).append(label)
    for disp, owners in shown.items():
        if len(owners) > 1:
            for i, l in enumerate(owners, 1):
                result[l]["display"] = f"{disp} {i}"
    return result
