"""Score ECHO and Gemini on the AMI meetings against the human annotations.

    .venv/Scripts/python eval/score.py                -> results/scores_dev.json
    AMI_SET=test .venv/Scripts/python eval/score.py   -> results/scores_test.json

Metrics (all against people's labels, never against another model):

  WER          words, Whisper's English normaliser on both sides
  DER          who-spoke-when, 0.25 s collar (pyannote.metrics)
  attribution  share of words credited to the right person. Each hypothesis word's
               time is looked up in the reference turns; words in silence or in
               overlapping speech are skipped. Hypothesis speakers are matched to
               reference people one-to-one by the most shared talk time (Hungarian).
  names        share of participants given their real first name. Key: AMI participant
               ids, names as introduced in the kickoff meetings.
  wrong names  names given that aren't the person's ("Speaker 1"-style placeholders don't count)
  roles        share of participants given their AMI role (PM, ID, UI, ME).
  ROUGE        summary vs the hand-written abstract; decisions vs the decisions block;
               action items vs the actions block (ROUGE-1 F1, stemmed).
"""

import json
import re
import sys
from pathlib import Path

import jiwer
import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment
from transformers import WhisperTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from ami import MEETINGS, SPLIT, abstractive, participants, reference_text, turns  # noqa: E402
from echo import text  # noqa: E402

RESULTS = HERE / "results"
SYSTEMS = ("echo", "gemini")
norm = WhisperTokenizer.from_pretrained("openai/whisper-small").normalize
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Dev names, from the people's own introductions in the human transcripts; test names in names_test.json.
NAMES = {"FEE005": "laura", "MEE006": "david", "MEE007": "craig", "MEE008": "andrew",
         "MEE009": "adam", "MEE010": "tarik", "MEE011": "dave", "MEE012": "ebenezer"}
NAMES.update(json.loads((HERE / "names_test.json").read_text(encoding="utf8")))
# Spellings of the same name (as heard by different speech models). "Andre" and "Greg" are other names.
SAME_NAME = {"dave": "david", "david": "david", "tarik": "tarik", "tariq": "tarik", "tarek": "tarik",
             "ebenezar": "ebenezer"}
ROLE_WORDS = {"PM": ("project manager", "manager", "chair", "lead", "pm"),
              "ID": ("industrial",), "UI": ("user interface", "interface", "ui"), "ME": ("marketing",)}
PLACEHOLDERS = re.compile(r"^(speaker|person|participant|member|unknown|team|attendee)\b|^person_\d", re.I)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else None


def first_name(s: str) -> str:
    """'Dave Cochran' -> 'david'. Placeholders like 'Speaker 1' or 'Person_2' are no name at all."""
    s = (s or "").strip()
    if not s or PLACEHOLDERS.search(s):
        return ""
    s = s.split(" ")[0].strip(".,()").lower()
    return SAME_NAME.get(s, s)


def role_of(s: str) -> str | None:
    s = (s or "").lower()
    for role, words in ROLE_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", s) for w in words):
            return role
    return None


def clock(stamp: str) -> float:
    total = 0.0
    for p in re.findall(r"\d+(?:\.\d+)?", stamp):
        total = total * 60 + float(p)
    return total


def annotation(items) -> Annotation:
    ann = Annotation()
    for start, end, spk in items:
        if end > start:
            ann[Segment(start, end)] = spk
    return ann


def match(hyp_turns, ref_turns) -> dict[str, str]:
    """Hypothesis label -> reference person, one-to-one, maximising shared talk time."""
    hyps = sorted({t[2] for t in hyp_turns})
    refs = sorted({t[2] for t in ref_turns})
    shared = np.zeros((len(hyps), len(refs)))
    for hs, he, h in hyp_turns:
        for rs, re_, r in ref_turns:
            o = min(he, re_) - max(hs, rs)
            if o > 0:
                shared[hyps.index(h), refs.index(r)] += o
    rows, cols = linear_sum_assignment(-shared)
    return {hyps[i]: refs[j] for i, j in zip(rows, cols) if shared[i, j] > 0}


def who_talks(ref_turns, t: float) -> str | None:
    """The one reference person talking at time t, or None for silence / overlap."""
    active = {s for a, b, s in ref_turns if a <= t < b}
    return next(iter(active)) if len(active) == 1 else None


def attribution(word_times: list[tuple[float, str]], mapping: dict[str, str], ref_turns) -> float:
    right = total = 0
    for t, spk in word_times:
        truth = who_talks(ref_turns, t)
        if truth is None:
            continue
        total += 1
        right += mapping.get(spk) == truth
    return right / total if total else 0.0


def spread(start: float, end: float, words: list[str], label: str) -> list[tuple[float, str]]:
    """A segment with no word timings: assume its words are evenly spaced."""
    n = len(words)
    return [(start + (end - start) * (i + 0.5) / n, label) for i in range(n)]


def identity(mapping: dict[str, str], said: dict[str, tuple[str, str]], people: dict) -> dict:
    """said = {hyp label: (name, role text)}. Scored per real participant."""
    back = {ref: hyp for hyp, ref in mapping.items()}
    names = roles = wrong = named_people = 0
    detail = {}
    for letter, (role, pid) in people.items():
        hyp = back.get(letter)
        name, role_text = said.get(hyp, ("", "")) if hyp else ("", "")
        truth = NAMES.get(pid)  # None: this person's name is never said in the series
        given = first_name(name)
        got_name = truth is not None and given == SAME_NAME.get(truth, truth)
        # A name that isn't theirs, including any name for someone whose name is never said.
        wrong += bool(given) and not got_name
        if truth is not None:
            named_people += 1
            names += got_name
        got_role = role_of(role_text) == role
        roles += got_role
        detail[letter] = {"truth": f"{truth or '(never said)'} ({role})", "said": f"{name or '-'} / {role_text or '-'}",
                          "name_ok": got_name, "role_ok": got_role}
    return {"names": names / named_people if named_people else None, "roles": roles / len(people),
            "wrong_names": wrong, "people": len(people), "detail": detail}


def r1(ref_sents: list[str], hyp_sents: list[str]) -> float | None:
    if not ref_sents:
        return None
    return rouge.score(" ".join(ref_sents), " ".join(hyp_sents) or "-")["rouge1"].fmeasure


def score_meeting(m: str) -> dict | None:
    ours = load(RESULTS / "echo" / f"{m}.json")
    if ours is None:
        return None
    ref_turns = turns(m)
    people = participants(m)
    dia = load(RESULTS / "diarize" / f"{m}.json")["turns"]
    words = load(RESULTS / "words" / f"{m}.json")["words"]
    gem = load(RESULTS / "gemini" / f"{m}.json")
    pyannote = [(t["start"], t["end"], t["speaker"]) for t in dia]
    mapping = match(pyannote, ref_turns)
    row = {"audio_minutes": ours["audio_minutes"]}

    # -- transcript and who spoke when
    ref_text = norm(reference_text(m))
    row["wer"] = {"echo": jiwer.wer(ref_text, norm(" ".join(w["word"] for w in words)))}
    row["der"] = {"echo": DiarizationErrorRate(collar=0.25)(annotation(ref_turns), annotation(pyannote))}
    labelled = text.assign_words(words, dia)
    row["attribution"] = {"echo": attribution([((w["start"] + w["end"]) / 2, w["speaker"]) for w in labelled],
                                              mapping, ref_turns)}

    # -- names and roles
    said = {s["label"]: (s["name"] if s["name"] != "unknown" else "", s["role"] if s["role"] != "unknown" else "")
            for s in ours["speakers"]}
    row["identity"] = {"echo": identity(mapping, said, people)}

    # -- minutes
    mins = ours["minutes"]
    abstract = " ".join(abstractive(m, "abstract"))
    ref_dec, ref_act = abstractive(m, "decisions"), abstractive(m, "actions")
    row["summary_rouge1"] = {"echo": rouge.score(abstract, mins["summary"])["rouge1"].fmeasure}
    row["decisions_rouge1"] = {"echo": r1(ref_dec, mins["decisions"])}
    row["actions_rouge1"] = {"echo": r1(ref_act, [f"{a['owner']} will {a['task']}" for a in mins["actions"]])}
    row["counts"] = {"ref_decisions": len(ref_dec), "echo_decisions": len(mins["decisions"]),
                     "ref_actions": len(ref_act), "echo_actions": len(mins["actions"])}

    if gem:
        end = max(e for _, e, _ in ref_turns)
        starts = [clock(t["start"]) for t in gem["transcript"]]
        g_turns = [(starts[i], max(starts[i + 1] if i + 1 < len(starts) else end, starts[i]), t["speaker"])
                   for i, t in enumerate(gem["transcript"])]
        g_map = match(g_turns, ref_turns)
        g_times = [x for (a, b, s), t in zip(g_turns, gem["transcript"]) for x in spread(a, b, t["text"].split(), s)]
        row["wer"]["gemini"] = jiwer.wer(ref_text, norm(" ".join(t["text"] for t in gem["transcript"])))
        row["der"]["gemini"] = DiarizationErrorRate(collar=0.25)(annotation(ref_turns), annotation(g_turns))
        row["attribution"]["gemini"] = attribution(g_times, g_map, ref_turns)
        # Gemini's speaker label is its name for the person ("Laura", "Speaker 2", "Project Manager").
        row["identity"]["gemini"] = identity(g_map, {s: (s, s) for s in g_map}, people)
        row["summary_rouge1"]["gemini"] = rouge.score(abstract, gem["summary"])["rouge1"].fmeasure
        row["decisions_rouge1"]["gemini"] = r1(ref_dec, gem.get("decisions", []))
        row["actions_rouge1"]["gemini"] = r1(ref_act, [f"{a['owner']} will {a['task']}" for a in gem.get("actions", [])])
        row["counts"].update(gemini_decisions=len(gem.get("decisions", [])), gemini_actions=len(gem.get("actions", [])),
                             gemini_model=gem.get("model"))
    row["echo_stage_seconds"] = ours["stage_seconds"]
    row["echo_llm"] = ours["llm"]
    return row


def main() -> None:
    rows = {m: r for m in MEETINGS if (r := score_meeting(m)) is not None}

    overall = {"split": SPLIT, "meetings": list(rows),
               "audio_hours": round(sum(r["audio_minutes"] for r in rows.values()) / 60, 2)}
    for system in SYSTEMS:
        o = {}
        for key in ("wer", "der", "attribution", "summary_rouge1", "decisions_rouge1", "actions_rouge1"):
            vals = [r[key][system] for r in rows.values() if isinstance(r[key].get(system), (int, float))]
            o[key] = sum(vals) / len(vals) if vals else None
        ids = [r["identity"][system] for r in rows.values() if system in r["identity"]]
        named = [i["names"] for i in ids if i["names"] is not None]
        o["names"] = sum(named) / len(named) if named else None
        o["roles"] = sum(i["roles"] for i in ids) / len(ids) if ids else None
        o["wrong_names"] = sum(i["wrong_names"] for i in ids)
        o["people"] = sum(i["people"] for i in ids)
        o["meetings"] = len(ids)
        overall[system] = o
    out = RESULTS / f"scores_{SPLIT}.json"
    out.write_text(json.dumps({"meetings": rows, "overall": overall}, indent=1), encoding="utf8")

    def f(v):
        return "   -  " if v is None else f"{v * 100:5.1f}%"

    keys = ("wer", "der", "attribution", "names", "summary_rouge1", "decisions_rouge1", "actions_rouge1")
    print(f"{'meeting':8} | " + " | ".join(f"{k[:13]:>13}" for k in keys) + "   (echo / gemini)")
    for m, r in rows.items():
        cells = [r[k] if k != "names" else {s: v["names"] for s, v in r["identity"].items()} for k in keys]
        print(f"{m:8} | " + " | ".join(f"{f(c.get('echo'))} {f(c.get('gemini'))}" for c in cells))
    print(json.dumps(overall, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
