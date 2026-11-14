# Evaluation

ECHO and a cloud reference, scored against human annotations from the
[AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (v1.6.2 manual annotations,
Mix-Headset audio, 4 participants per meeting, scenario meetings about designing a remote control).

| System | What it is |
|---|---|
| **ECHO** | this repository: faster-whisper small int8, pyannote 3.1, word-level alignment, Qwen2.5-3B (llama.cpp) for speakers and minutes with every claim checked against the transcript |
| **Gemini** | a cloud reference: one Gemini audio call per recording returning the same minutes format as JSON (`run_gemini.py`, using the portfolio site's Gemini backend) |

## Dev and test

| Split | Meetings | Use |
|---|---|---|
| dev | ES2002a/b/c, ES2003a/b/c | every rule was developed while looking at these |
| test | ES2004a/c, ES2005a/c (1.7 h) | held out: not looked at while building. The code was frozen and run on them once |

Dev numbers are optimistic by construction; **the test numbers are the ones to quote.**
Test audio is downloaded from the AMI mirror into `data/ami_test/` (not in git):
`https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<meeting>/audio/<meeting>.Mix-Headset.wav`.

## Results

| Metric | Test: ECHO | Test: Gemini | Dev: ECHO | Dev: Gemini |
|---|---|---|---|---|
| Attribution ↑ | **99.0%** | 81.6% | **98.7%** | 85.7% |
| DER ↓ | **21.6%** | 42.0% | **13.2%** | 42.6% |
| WER ↓ | 29.1% | **25.8%** | **23.3%** | 24.4% |
| Names ↑ | 12.5% | **50.0%** | 20.8% | **54.2%** |
| Wrong names ↓ | **0** | 1 | **1** | 4 |
| Roles ↑ | **6.2%** | 0.0% | **20.8%** | 0.0% |
| Summary ROUGE-1 ↑ | 20.6% | **32.9%** | 22.6% | **29.6%** |
| Decisions ROUGE-1 ↑ | 9.1% | **14.9%** | 18.5% | **26.9%** |
| Actions ROUGE-1 ↑ | 22.0% | **37.6%** | 14.3% | **29.3%** |

Per meeting, per person and per stage: `results/scores_test.json`, `results/scores_dev.json`.

## Metrics

| Metric | Meaning | Better |
|---|---|---|
| WER | word error rate of the transcript; both sides through Whisper's English normaliser (fillers removed, "twenty five" = "25") | lower |
| DER | diarization error rate, pyannote.metrics, 0.25 s collar, overlap included | lower |
| Attribution | share of words credited to the right person. Each word's time is looked up in the human speaker turns (silence and overlapping speech skipped); system speakers are matched to people one-to-one by shared talk time (Hungarian) | higher |
| Names | share of people whose name is said in the meeting series who get that name | higher |
| Wrong names | people given a name that isn't theirs, including anyone whose name is never said. Placeholders ("Speaker 1", "Person_2") are not names | lower |
| Roles | share of people given their AMI role (project manager, industrial designer, user interface, marketing) | higher |
| Summary / decisions / actions | ROUGE-1 F1 (stemmed) against the hand-written abstract, decisions and actions blocks of the AMI abstractive summary | higher |

Name key: AMI participant ids are stable across a meeting series; names come from the human
transcripts (self-introductions, or the person who answers when addressed). Dev: in `score.py`;
test: `names_test.json`. Two test names (William, Brian) are inferred from who answers when
addressed, so they are "likely", not certain.

## Files

```
ami.py            reads the AMI annotations (words, speaker turns, abstractive summaries)
run.py            runs ECHO on a split, saving transcript, diarization and output separately
run_gemini.py     runs the Gemini reference (needs the portfolio backend and a GEMINI_API_KEY)
score.py          scores everything in results/ -> results/scores_<split>.json
results/words     faster-whisper words with times, per meeting
results/diarize   pyannote speaker turns, per meeting
results/echo      ECHO's speakers, minutes and timings, per meeting
results/gemini    Gemini's output, per meeting
```

## Run it

```bash
# ECHO on dev (needs the AMI corpus, path in ami.py) or test (needs data/ami_test)
.venv/Scripts/python eval/run.py
AMI_SET=test .venv/Scripts/python eval/run.py
# score
.venv/Scripts/python eval/score.py                 # -> results/scores_dev.json
AMI_SET=test .venv/Scripts/python eval/score.py    # -> results/scores_test.json
```

Scoring also needs `jiwer`, `rouge-score`, `scipy` and `pyannote.metrics`. Run one meeting per
process with `ECHO_THREADS=4` on a 16 GB laptop: an all-at-once 8-thread run ran out of memory.

## Caveats

- Gemini's model varies: the backend falls back through a list when a model is busy (503). Dev used
  `gemini-flash-latest` throughout; test used `gemini-flash-lite-latest` (ES2004a) and
  `gemini-3.5-flash` (others). See each `results/gemini/*.json` `model` field.
- ROUGE rewards word overlap with one annotator's wording, not correctness.
- AMI is scenario role-play (one company, one product). Names are said rarely after the kickoff
  meeting, which caps every system's name score.
