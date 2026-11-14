"""The verifier, with the LLM replaced by a fixed answer: what survives and what doesn't."""
from echo import speakers

LINES = [
    ("00:01", "S0", "Okay so this is the kick-off meeting. I'm Laura and I'm the project manager."),
    ("00:09", "S1", "Hi I'm David and I'm supposed to be an industrial designer."),
    ("00:15", "S2", "I'm Craig and I'm User Interface."),
    ("00:20", "S3", "Mm-hmm. Right, let's look at the budget."),
]


def run(monkeypatch, proposed):
    monkeypatch.setattr(speakers.llm, "json_call", lambda *a, **k: proposed)
    return speakers.identify(LINES)


def test_keeps_verified_names_and_reads_role_from_own_quote(monkeypatch):
    out = run(monkeypatch, {
        "S0": {"name": "Laura", "role": "project manager", "evidence": "I'm Laura and I'm the project manager"},
        "S1": {"name": "David", "role": "unknown", "evidence": "I'm David and I'm supposed to be an industrial designer"},
        "S2": {"name": "Craig", "role": "unknown", "evidence": "I'm Craig and I'm User Interface"},
        "S3": {"name": "unknown", "role": "unknown", "evidence": ""},
    })
    assert out["S0"]["display"] == "Laura" and out["S0"]["role"] == "project manager"
    assert out["S1"]["role"] == "industrial designer"  # model said unknown; the quote says it
    assert out["S2"]["role"] == "user interface"
    assert out["S3"]["display"] == "Person_1" and not out["S3"]["verified"]


def test_rejects_invented_name_and_fake_quote(monkeypatch):
    out = run(monkeypatch, {
        "S0": {"name": "Laura", "role": "", "evidence": "I'm Laura"},
        "S1": {"name": "Priya", "role": "unknown", "evidence": "I'm Priya the engineer"},
        "S2": {"name": "Craig", "role": "", "evidence": ""},
        "S3": {"name": "unknown", "role": "", "evidence": ""},
    })
    # The invented name is thrown out, and S1's own "I'm David" takes its place.
    assert out["S1"]["name"] == "David"
    assert "quote not in transcript" in out["S1"]["rejected"] and "name 'Priya' never said" in out["S1"]["rejected"]


def test_someone_elses_introduction_is_not_yours(monkeypatch):
    out = run(monkeypatch, {
        "S0": {"name": "unknown", "role": "", "evidence": ""},
        "S1": {"name": "unknown", "role": "", "evidence": ""},
        "S2": {"name": "unknown", "role": "", "evidence": ""},
        "S3": {"name": "Laura", "role": "", "evidence": "I'm Laura and I'm the project manager"},
    })
    assert out["S3"]["name"] == "unknown"
    assert "introduction belongs to another speaker" in out["S3"]["rejected"]


def test_one_name_one_speaker(monkeypatch):
    out = run(monkeypatch, {
        "S0": {"name": "Laura", "role": "", "evidence": "I'm Laura"},
        "S1": {"name": "unknown", "role": "", "evidence": ""},
        "S2": {"name": "unknown", "role": "", "evidence": ""},
        "S3": {"name": "Laura", "role": "", "evidence": "let's look at the budget"},
    })
    assert out["S0"]["name"] == "Laura" and out["S3"]["name"] == "unknown"


def test_shortlist_keeps_cues_and_first_lines():
    lines = [("00:00", "A", "hello")] + [("00:01", "B", f"filler {i}") for i in range(10)] + \
            [("00:05", "B", "thanks Laura"), ("00:06", "A", "no problem")]
    s = speakers.shortlist(lines)
    assert "thanks Laura" in s and "no problem" in s and "filler 9" in s and "filler 5" not in s  # 9 is the line before the cue


def test_fallbacks_read_name_and_role_from_own_words(monkeypatch):
    lines = LINES[:3] + [("00:20", "S3", "Right. " + "We should talk about it. " * 30 + "Oh and I'm Andrew, I'm the marketing expert.")]
    monkeypatch.setattr(speakers.llm, "json_call", lambda *a, **k: {
        l: {"name": "unknown", "role": "unknown", "evidence": ""} for l in ("S0", "S1", "S2", "S3")})
    out = speakers.identify(lines)
    assert out["S1"]["name"] == "David" and out["S1"]["role"] == "industrial designer"
    assert out["S3"]["name"] == "Andrew" and out["S3"]["role"] == "marketing expert"
    assert "I'm Andrew" in speakers.shortlist(lines)  # late in a long turn, still shown to the model


def test_roll_call_answer_counts_as_evidence(monkeypatch):
    lines = [("00:02", "B", "My name's Adam. And who are you?"), ("00:34", "D", "Tarik Rahman. T-A-R-I-K."),
             ("00:36", "B", "And your role in this is"), ("00:49", "A", "Uh, Dave Cochrane."),
             ("00:53", "B", "That's David, Andrew and Craig isn't it")]
    monkeypatch.setattr(speakers.llm, "json_call", lambda *a, **k: {
        "A": {"name": "Dave", "role": "", "evidence": "Uh, Dave Cochrane."},
        "B": {"name": "Adam", "role": "", "evidence": "My name's Adam."},
        "D": {"name": "Tarik", "role": "", "evidence": "Tarik Rahman."}})
    out = speakers.identify(lines)
    assert [out[k]["name"] for k in "ABD"] == ["Dave", "Adam", "Tarik"]
    # "Craig" is only mentioned in a list, never by or to a speaker: it must not stick to anyone.
    assert speakers._name_supported("Craig", "B", lines)[0] is False


def fixed(monkeypatch, answer):
    monkeypatch.setattr(speakers.llm, "json_call", lambda *a, **k: answer)


def test_generic_and_unsupported_roles_are_dropped(monkeypatch):
    # ES2002b/c, ES2003b: "Team Member 1..4", "Participant 3", "Speaker 6" told nobody anything
    lines = [("00:01", "A", "Okay let's start with the agenda."), ("00:05", "B", "Sounds good to me.")]
    fixed(monkeypatch, {"A": {"name": "unknown", "role": "team leader", "evidence": ""},
                        "B": {"name": "unknown", "role": "team member", "evidence": ""}})
    out = speakers.identify(lines)
    assert out["A"]["role"] == "unknown" and "role 'team leader' not stated by or to this speaker" in out["A"]["rejected"]
    assert out["B"]["role"] == "unknown" and out["B"]["display"].startswith("Person_")


def test_role_given_by_the_chair_counts(monkeypatch):
    # ES2003a: "And your role in this is Industrial Designer" said right after Tarik's turn
    lines = [("00:34", "D", "Tarek Raman, T-A-R-I-K."), ("00:36", "B", "And your role in this is Industrial Designer.")]
    fixed(monkeypatch, {"B": {"name": "unknown", "role": "", "evidence": ""},
                        "D": {"name": "Tarek", "role": "industrial designer", "evidence": "Tarek Raman"}})
    out = speakers.identify(lines)
    assert out["D"]["name"] == "Tarek" and out["D"]["role"] == "industrial designer"


def test_roll_call_from_the_opening_of_a_long_turn(monkeypatch):
    # ES2003a: Ebenezer's answer merged with later talk, so the whole turn was too long
    lines = [("00:02", "B", "Right, my name is Adam. We're here for the kick-off."),
             ("00:13", "C", "Ebenezer at Demisoy. Do you want me to spell it? I am the marketing expert, by the way."),
             ("00:24", "B", "Ebenezer. And your role is?")]  # the chair repeating it is the corroboration
    fixed(monkeypatch, {"B": {"name": "Adam", "role": "", "evidence": "my name is Adam"},
                        "C": {"name": "unknown", "role": "chairman of the meeting",
                              "evidence": "[00:02] SPEAKER_00: Right, my name is Adam."}})
    out = speakers.identify(lines)
    assert out["C"]["name"] == "Ebenezer" and out["C"]["role"] == "marketing expert"
    assert speakers._roll_call("Craig, what do you think?") is None  # being addressed, not answering


def test_addressee_votes_name_the_quiet_speaker(monkeypatch):
    # ES2002b: nobody introduced themselves, but "what do you think Craig?" was answered twice by the same person
    lines = [("00:01", "A", "Speech recognition is something to consider, what do you think Craig?"),
             ("00:05", "C", "It could work if it's cheap."), ("00:09", "A", "Great. Do you want to recap on that Craig?"),
             ("00:12", "C", "Sure, three functions."), ("00:20", "A", "great, thanks")]
    fixed(monkeypatch, {l: {"name": "unknown", "role": "", "evidence": ""} for l in "AC"})
    out = speakers.identify(lines)
    assert out["C"]["name"] == "Craig" and out["C"]["named_by"] == "addressed 2x"
    assert out["A"]["name"] == "unknown"  # "Great" is not a name


def test_everyday_words_are_not_roll_call_names(monkeypatch):
    # An earlier build named an ES2002a speaker "You": a turn opened "You know." and others also say "You..."
    lines = [("00:01", "A", "I'm Greg and I do the interface. You need to see this, you will like it."),
             ("00:04", "B", "Yeah I am. You know, it's fine.")]
    fixed(monkeypatch, {"A": {"name": "Greg", "role": "", "evidence": "I'm Greg"},
                        "B": {"name": "Greg", "role": "", "evidence": "I'm Greg and I do the interface."}})
    out = speakers.identify(lines)
    assert out["B"]["name"] == "unknown"  # "I am. You know" is not "I am You"
    assert out["B"]["evidence"] == ""     # and A's introduction is not B's evidence


def test_one_word_turns_are_not_roll_calls(monkeypatch):
    # ES2002a, real lines from the speaker who was named "You"
    lines = [("01:17", "A", "I'm an industrial designer."), ("04:41", "A", "You"), ("07:23", "A", "probably"),
             ("04:40", "B", "You can draw anything. Do you see it?")]
    fixed(monkeypatch, {l: {"name": "unknown", "role": "", "evidence": ""} for l in "AB"})
    out = speakers.identify(lines)
    assert out["A"]["name"] == "unknown" and out["A"]["role"] == "industrial designer"
    assert speakers._roll_call("probably") is None and speakers._roll_call("Uh, Dave Cochrane.") == "Dave"
