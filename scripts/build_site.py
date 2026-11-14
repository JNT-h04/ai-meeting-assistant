"""Build the static showcase site in docs/ from ECHO's saved evaluation output.

    .venv/Scripts/python scripts/build_site.py

Nothing is generated or edited here: every transcript line, name, quote, decision and score
is copied from eval/results (ECHO's real output on the held-out AMI meetings and the scorer's
comparison with the human annotations). The meeting audio (AMI, CC BY 4.0) is re-encoded to
MP3 so the page can play each moment it cites.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

import ami  # noqa: E402
import score  # noqa: E402

RESULTS = ROOT / "eval" / "results"
SITE = ROOT / "docs"
TEST = ["ES2004a", "ES2004c", "ES2005a", "ES2005c"]
ROLES = {"PM": "project manager", "ID": "industrial designer", "UI": "user interface designer", "ME": "marketing expert"}
METRICS = ("attribution", "der", "wer", "names", "wrong_names", "roles", "summary_rouge1", "decisions_rouge1",
           "actions_rouge1")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def truth_by_label(meeting: str) -> dict[str, dict]:
    """ECHO's speaker label -> the real person (AMI annotation) it was matched to by the scorer."""
    dia = load(RESULTS / "diarize" / f"{meeting}.json")["turns"]
    mapping = score.match([(t["start"], t["end"], t["speaker"]) for t in dia], ami.turns(meeting))
    people = ami.participants(meeting)
    out = {}
    for label, letter in mapping.items():
        role, pid = people[letter]
        out[label] = {"person": letter, "name": (score.NAMES.get(pid) or "").title() or None, "role": ROLES[role]}
    return out


def encode_audio(meeting: str) -> str:
    dst = SITE / "audio" / f"{meeting}.mp3"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        audio, sr = sf.read(ami.audio_path(meeting), dtype="float32", always_2d=True)
        with sf.SoundFile(dst, "w", sr, 1, format="MP3", subtype="MPEG_LAYER_III") as f:
            f.write(audio.mean(axis=1))
    return f"audio/{meeting}.mp3"


def meeting(m: str, scored: dict) -> dict:
    r = load(RESULTS / "echo" / f"{m}.json")
    mins = r["minutes"]
    truth = truth_by_label(m)
    s = scored["meetings"][m]
    speakers = []
    for sp in r["speakers"]:
        t = truth.get(sp["label"])
        name_ok = bool(t and t["name"] and sp["name"] != "unknown" and score.first_name(sp["name"]) ==
                       score.SAME_NAME.get(t["name"].lower(), t["name"].lower()))
        speakers.append({**sp, "truth": t, "name_ok": name_ok})
    return {
        "id": m,
        "minutes_long": r["audio_minutes"],
        "audio": encode_audio(m),
        "title": mins["title"],
        "summary": mins["summary"],
        "topics": mins["topics"],
        "decisions": [{"text": d, "at": a} for d, a in zip(mins["decisions"], mins["decisions_at"])],
        "actions": [{**a, "at": at} for a, at in zip(mins["actions"], mins["actions_at"])],
        "open_questions": mins["open_questions"],
        "dropped_unverified": mins.get("dropped_unverified", 0),
        "speakers": speakers,
        "transcript": [{k: t[k] for k in ("label", "speaker", "start", "start_s", "end_s", "text")}
                       for t in mins["transcript"]],
        "scores": {sys_: {k: (s["identity"][sys_][k] if k in ("names", "wrong_names", "roles") else s[k].get(sys_))
                          for k in METRICS if sys_ in s["identity"]}
                   for sys_ in ("echo", "gemini")},
        "reference": {"decisions": s["counts"]["ref_decisions"], "actions": s["counts"]["ref_actions"]},
        "stage_seconds": r["stage_seconds"],
        "llm": r["llm"],
    }


def main() -> None:
    scored = load(RESULTS / "scores_test.json")
    overall = scored["overall"]
    data = {
        "source": "eval/results (ECHO's saved output) and eval/results/scores_test.json (scored against AMI annotations)",
        "overall": {sys_: {k: overall[sys_][k] for k in METRICS} for sys_ in ("echo", "gemini")},
        "audio_hours": overall["audio_hours"],
        "meetings": [meeting(m, scored) for m in TEST],
    }
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    out = SITE / "data" / "meetings.json"
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf8")
    print(f"{out}: {out.stat().st_size / 1e3:.0f} kB; audio: "
          f"{sum(p.stat().st_size for p in (SITE / 'audio').glob('*.mp3')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
