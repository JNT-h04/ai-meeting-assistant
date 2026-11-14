from echo import text


def seg(start, end, words, speaker=None):
    s = {"start": start, "end": end, "text": words}
    if speaker:
        s["speaker"] = speaker
    return s


def test_drop_repeats_keeps_two_copies():
    segs = [seg(i, i + 1, "The last few years.") for i in range(24)] + [seg(30, 31, "Next.")]
    out = text.drop_repeats(segs)
    assert [s["text"] for s in out] == ["The last few years.", "The last few years.", "Next."]


def test_assign_speakers_sums_overlap_per_speaker():
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 1.6, "speaker": "B"},
        {"start": 1.6, "end": 2.2, "speaker": "B"},
        {"start": 2.2, "end": 3.0, "speaker": "A"},
    ]
    # A overlaps 0.5 + 0.3 = 0.8, B overlaps 1.2: B wins only because its turns are summed.
    out = text.assign_speakers([seg(0.5, 2.5, "hi")], turns)
    assert out[0]["speaker"] == "B"


def test_assign_speakers_unknown_without_overlap():
    assert text.assign_speakers([seg(5, 6, "hi")], [{"start": 0, "end": 1, "speaker": "A"}])[0]["speaker"] == "UNKNOWN"


def test_names_need_a_capital_and_refuse_common_words():
    assert text.name_from_intro("Hi everyone, my name is Laura and I'm the PM") == "Laura"
    assert text.name_from_intro("I am going to start") is None  # the notebook said "Going"
    assert text.name_from_intro("I'm Sure that works") is None
    assert text.name_from_intro("this is david") is None


def test_name_speakers_falls_back_in_speaking_order():
    segs = [seg(0, 1, "Okay let's start", "SPEAKER_02"), seg(1, 2, "Hi, I'm Craig", "SPEAKER_00"),
            seg(2, 3, "Fine", "SPEAKER_01"), seg(3, 4, "I'm Craig too", "SPEAKER_01")]
    assert text.name_speakers(segs) == {"SPEAKER_02": "Person_1", "SPEAKER_00": "Craig", "SPEAKER_01": "Person_2"}


def test_merge_turns_joins_same_speaker():
    out = text.merge_turns([seg(0, 1, "a", "X"), seg(1, 2, "b", "X"), seg(2, 3, "c", "Y")])
    assert out == [seg(0, 2, "a b", "X"), seg(2, 3, "c", "Y")]


def test_word_chunks_overlap_and_no_redundant_tail():
    words = " ".join(str(i) for i in range(500))
    chunks = text.word_chunks(words, 400, 50)
    assert len(chunks) == 2
    assert chunks[1].split()[0] == "350" and chunks[1].split()[-1] == "499"
    assert text.word_chunks("", 400, 50) == []


def test_passages_windows():
    lines = [f"l{i}" for i in range(7)]
    assert text.passages(lines) == ["l0\nl1\nl2\nl3\nl4", "l3\nl4\nl5\nl6"]


def test_clock():
    assert text.clock(0) == "00:00" and text.clock(125.9) == "02:05"


def w(start, end, word):
    return {"start": start, "end": end, "word": word}


def test_assign_words_splits_a_segment_at_the_speaker_change():
    turns = [{"start": 0.0, "end": 1.9, "speaker": "A"}, {"start": 2.0, "end": 4.0, "speaker": "B"}]
    words = [w(0.1, 0.5, "Okay,"), w(0.6, 1.0, "thanks."), w(2.1, 2.4, "Sure,"), w(2.5, 3.0, "so")]
    segs = text.words_to_segments(text.assign_words(words, turns))
    assert [(s["speaker"], s["text"]) for s in segs] == [("A", "Okay, thanks."), ("B", "Sure, so")]


def test_assign_words_gap_snaps_to_nearest_then_keeps_previous():
    turns = [{"start": 0.0, "end": 1.0, "speaker": "A"}, {"start": 10.0, "end": 11.0, "speaker": "B"}]
    out = text.assign_words([w(1.3, 1.5, "near"), w(5.0, 5.2, "far")], turns)
    assert [x["speaker"] for x in out] == ["A", "A"]


def test_words_to_segments_breaks_on_long_pause():
    words = [{**w(0, 0.5, "one"), "speaker": "A"}, {**w(3.0, 3.5, "two"), "speaker": "A"}]
    assert len(text.words_to_segments(words, pause=1.5)) == 2
