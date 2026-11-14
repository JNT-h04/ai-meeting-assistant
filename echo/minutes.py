"""Minutes from a named transcript: title, summary, topics, decisions, action items, open questions.

The categories follow the AMI corpus's own hand-written summaries (abstract, decisions,
actions, problems), so the output can be scored against them.

Extract then verify (the same idea as speakers.py):
  1. extract  the meeting is read in ~1,800-token sections. For each, the LLM lists
              decisions, actions and open questions, each with the [mm:ss] of the line
              where it was said. A small model finds far more in a short section than
              in a whole meeting at once (one whole-meeting call found 0 of ES2002a's 3 decisions).
  2. verify   code keeps an item only if its timestamp is a real line in that section
              and the item shares content words with that line or its neighbours.
              Duplicates across sections are merged.
  3. write    one more call writes the title, summary and topics from the section
              notes. Decisions and actions stay the verified, cited items.
"""

import re
from difflib import SequenceMatcher

from echo import llm

SECTION_TOKENS = 1800
MAX_ITEMS = 10
_CLOCK = re.compile(r"^\[(\d\d:\d\d)\]")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the and or but if so to of in on at for with by from as is are was were be been being it its this
that these those we you they i he she them us our your their my me will would should could can may might must do does
did have has had not no yes yeah okay ok um uh just like really very going gonna get got make made need about all also
there here what which who when where how then than too more most some any one two""".split())

ITEM = {"type": "object", "properties": {"text": {"type": "string"}, "at": {"type": "string"}},
        "required": ["text", "at"]}
ACTION = {"type": "object", "properties": {
    "task": {"type": "string"}, "owner": {"type": "string"}, "due": {"type": "string"}, "at": {"type": "string"}},
    "required": ["task", "owner", "due", "at"]}
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "decisions": {"type": "array", "items": ITEM, "maxItems": 6},
        "actions": {"type": "array", "items": ACTION, "maxItems": 6},
        "open_questions": {"type": "array", "items": ITEM, "maxItems": 4},
    },
    "required": ["points", "decisions", "actions", "open_questions"],
}
EXTRACT_SYSTEM = """You take notes on one section of a meeting transcript. Lines look like "[mm:ss] Name: words".
points: what was discussed, as short sentences.
decisions: anything the group agreed on, chose, or was told is fixed: a price, budget, target, deadline,
  requirement or design choice. Example: "[07:28] Ann: we'll sell it for 25 euro" -> {"text": "The product will sell for 25 euro", "at": "07:28"}.
actions: tasks someone will do after this point. owner = the speaker name exactly as written, or "unassigned".
  due = the deadline if one was said, else "not stated".
open_questions: issues raised but not settled.
at = the [mm:ss] of the line where it is said, copied exactly. Only include what is in this section; empty lists are fine."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
    "required": ["title", "summary", "topics"],
}
SUMMARY_SYSTEM = ("You write the opening of meeting minutes from notes taken in order through one meeting. "
                  "title: at most 8 words. summary: 3 to 5 plain sentences on what the meeting was for, what was "
                  "decided and what happens next. topics: short phrases. Use only the notes.")


def sections(lines: list[str], budget: int = SECTION_TOKENS) -> list[list[str]]:
    out, cur, used = [], [], 0
    for line in lines:
        n = llm.tokens(line)
        if cur and used + n > budget:
            out.append(cur)
            cur, used = [], 0
        cur.append(line)
        used += n
    if cur:
        out.append(cur)
    return out


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def _grounded(item_text: str, at: str, part: list[str]) -> str | None:
    """The clock if `at` is a line in this section whose neighbourhood shares content words with the item."""
    at = at.strip("[] ")
    clocks = [(_CLOCK.match(l).group(1) if _CLOCK.match(l) else None) for l in part]
    if at not in clocks:
        return None
    i = clocks.index(at)
    near = " ".join(part[max(0, i - 1): i + 2])
    return at if len(_content(item_text) & _content(near)) >= 2 else None


def _dedupe(items: list[dict], key: str) -> list[dict]:
    kept: list[dict] = []
    for it in items:
        if any(SequenceMatcher(None, it[key].lower(), k[key].lower()).ratio() > 0.6 for k in kept):
            continue
        kept.append(it)
    return kept[:MAX_ITEMS]


def write(lines: list[str], people: list[str], on_progress=lambda done, total: None) -> dict:
    """lines = '[mm:ss] Name: text'. Returns the minutes plus how they were made."""
    known = {p.lower(): p for p in people}
    parts = sections(lines)
    notes, decisions, actions, questions = [], [], [], []
    dropped = 0
    for n, part in enumerate(parts):
        on_progress(n, len(parts) + 1)
        got = llm.json_call(EXTRACT_SYSTEM, "\n".join(part), EXTRACT_SCHEMA, max_tokens=700)
        notes.append(f"Section {n + 1} of {len(parts)}:\n" + "\n".join(f"- {p}" for p in got["points"]))
        for d in got["decisions"]:
            at = _grounded(d["text"], d["at"], part)
            if at:
                decisions.append({"text": d["text"].strip(), "at": at})
                notes[-1] += f"\nDecided: {d['text']}"
            else:
                dropped += 1
        for a in got["actions"]:
            at = _grounded(a["task"], a["at"], part)
            if at:
                actions.append({"task": a["task"].strip(), "owner": known.get(a["owner"].strip().lower(), "unassigned"),
                                "due": a["due"].strip() or "not stated", "at": at})
                notes[-1] += f"\nTo do: {a['task']}"
            else:
                dropped += 1
        for q in got["open_questions"]:
            at = _grounded(q["text"], q["at"], part)
            if at:
                questions.append({"text": q["text"].strip(), "at": at})
            else:
                dropped += 1
    on_progress(len(parts), len(parts) + 1)
    head = llm.json_call(SUMMARY_SYSTEM, "\n\n".join(notes), SUMMARY_SCHEMA, max_tokens=400)
    on_progress(len(parts) + 1, len(parts) + 1)
    decisions, actions, questions = _dedupe(decisions, "text"), _dedupe(actions, "task"), _dedupe(questions, "text")
    return {
        "title": head["title"].strip(),
        "summary": head["summary"].strip(),
        "topics": [t.strip() for t in head["topics"] if t.strip()],
        "decisions": [d["text"] for d in decisions],
        "decisions_at": [d["at"] for d in decisions],
        "actions": [{k: a[k] for k in ("task", "owner", "due")} for a in actions],
        "actions_at": [a["at"] for a in actions],
        "open_questions": [q["text"] for q in questions],
        "method": "extract-verify",
        "sections": len(parts),
        "dropped_unverified": dropped,
    }
