"""Extract-then-verify minutes, with the LLM replaced by fixed answers."""
from echo import minutes

LINES = [
    "[07:28] Laura: so according to the brief we're gonna be selling this remote control for 25 euro",
    "[07:40] Laura: and we're selling it on an international scale",
    "[08:19] Andrew: is that the wholesale price or the retail price",
    "[09:00] Laura: David, can you work on the working design for next time",
]


def run(monkeypatch, extract):
    answers = iter([extract, {"title": "Kick-off", "summary": "They met.", "topics": ["budget"]}])
    monkeypatch.setattr(minutes.llm, "json_call", lambda *a, **k: next(answers))
    monkeypatch.setattr(minutes.llm, "tokens", lambda text: len(text.split()))
    return minutes.write(LINES, ["Laura", "Andrew", "David"])


def test_keeps_grounded_items_with_their_timestamps(monkeypatch):
    out = run(monkeypatch, {
        "points": ["budget"],
        "decisions": [{"text": "The remote will sell for 25 euro", "at": "07:28"},
                      {"text": "It will be sold internationally on an international scale", "at": "[07:40]"}],
        "actions": [{"task": "Work on the working design", "owner": "david", "due": "next time", "at": "09:00"}],
        "open_questions": [{"text": "Is the price wholesale or retail?", "at": "08:19"}],
    })
    assert out["decisions"] == ["The remote will sell for 25 euro", "It will be sold internationally on an international scale"]
    assert out["decisions_at"] == ["07:28", "07:40"]
    assert out["actions"] == [{"task": "Work on the working design", "owner": "David", "due": "next time"}]
    assert out["open_questions"] == ["Is the price wholesale or retail?"]
    assert out["dropped_unverified"] == 0 and out["method"] == "extract-verify"


def test_drops_items_the_transcript_does_not_back(monkeypatch):
    out = run(monkeypatch, {
        "points": [],
        "decisions": [{"text": "The team chose a solar-powered battery", "at": "07:28"},   # wrong content
                      {"text": "The remote will sell for 25 euro", "at": "12:00"}],         # no such line
        "actions": [{"task": "Order samples from the factory", "owner": "Craig", "due": "", "at": "09:00"}],
        "open_questions": [],
    })
    assert out["decisions"] == [] and out["actions"] == [] and out["dropped_unverified"] == 3


def test_duplicates_across_sections_merge(monkeypatch):
    out = run(monkeypatch, {
        "points": [],
        "decisions": [{"text": "The remote will sell for 25 euro", "at": "07:28"},
                      {"text": "The remote will sell for 25 euros", "at": "07:28"}],
        "actions": [], "open_questions": [],
    })
    assert len(out["decisions"]) == 1
