# ECHO: meeting intelligence on open models

Upload a meeting recording. ECHO returns **who said what**, a summary, the **decisions**, the
**action items** (owner and due date), open questions, and answers to questions about the meeting,
each one pointing at the moment in the recording where it was said.

It runs entirely on open models, on a CPU, with no paid API. A small language model does the
understanding, and **every name, role, decision and action item it proposes is checked against the
transcript by code before it is shown**.

```
audio ─► faster-whisper ─► pyannote 3.1 ─► word-level ─► Qwen2.5-3B + verifier ─► LangChain RAG
         words + times     who spoke when   alignment     names, roles, minutes     cited answers
```

## Results

Measured against human annotations of the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
on **4 held-out meetings (1.7 hours)** that were never looked at while the system was built; the code
was frozen before the run. Gemini is the cloud reference: one Gemini audio call per recording,
returning the same minutes format.

| Metric | ECHO | Gemini (cloud) |
|---|---|---|
| Words credited to the right speaker ↑ | **99.0%** | 81.6% |
| Who-spoke-when error (DER) ↓ | **21.6%** | 42.0% |
| Wrong speaker names given ↓ | **0** | 1 |
| Speaker roles identified ↑ | **6.2%** | 0.0% |
| Transcript word error (WER) ↓ | 29.1% | **25.8%** |
| Names identified (of people whose name is said) ↑ | 12.5% | **50.0%** |
| Summary vs human abstract (ROUGE-1) ↑ | 20.6% | **32.9%** |
| Decisions vs human list (ROUGE-1) ↑ | 9.1% | **14.9%** |
| Action items vs human list (ROUGE-1) ↑ | 22.0% | **37.6%** |

**What it says:** ECHO is clearly better at *who said what*: half the diarization error and 99% of
words on the right person. It is conservative with names (it says "unknown" rather than guess) and
never named anyone wrongly on unseen meetings. A frontier cloud model is still better at
transcription and at writing summaries; ECHO's 3-billion-parameter model runs on a laptop-class CPU.

Speed on 2 CPU cores: about real time, 5 minutes of audio in about 5 minutes, three-quarters of it
in diarization. Peak memory: 3.8 GB.

Full method, every metric, dev-set numbers and per-meeting results: [`eval/README.md`](eval/README.md).

**Explore the results:** <https://jnt-h04.github.io/ai-meeting-assistant/> replays ECHO's real output on the four test
meetings with the audio, every speaker's evidence and the human ground truth (recorded results, not a live run).

## How it works

| Step | Model | What it does |
|---|---|---|
| Speech to text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `small`, int8 | Whisper on CTranslate2 with a voice-activity filter; every word gets a start and end time |
| Who spoke when | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | segmentation network + WeSpeaker voice embeddings + clustering |
| Alignment | (code) | each **word** goes to the speaker turn it overlaps most; the transcript is cut where the speaker changes |
| Speakers | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) Q4_K_M via llama.cpp | proposes each speaker's name, role and a supporting quote, as schema-constrained JSON |
| Minutes | same model | reads the meeting in ~1,800-token sections; lists decisions, action items and open questions with the `[mm:ss]` they were said; writes title, summary, topics |
| Questions | LangChain: BM25 + FAISS ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) fused, then the LLM | answers with cited timestamps, or says it wasn't discussed |

### Verified, not trusted

The language model is small, so it is asked for **evidence**, and ordinary code decides whether the
evidence holds (`echo/speakers.py`, `minutes.py`, `rag.py`):

- **Names** are kept only if tied to that speaker: their own introduction, their answer to a roll call
  ("Tarek Raman, T-A-R-I-K"), or someone addressing them right next to their turn. A quote that isn't in
  the transcript, someone else's introduction, or a second speaker with the same name is rejected.
- **Addressee voting** (`address.py`): every "…what do you think, Craig?" is a vote for whoever speaks
  next. Enough votes and a clear majority name a speaker who never introduced themselves.
- **Roles** must be said by or to the speaker; vague labels ("team member") are dropped.
- **Decisions and action items** must cite a real line of their section and share content words with it.
- **Answers** may only cite times the retriever actually returned.

Every rejection is kept and shown, so the page can say *why* someone is "Person_2".

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
set HF_TOKEN=hf_...        # an account that accepted the pyannote/speaker-diarization-3.1 terms
.venv/Scripts/uvicorn echo.api:app --port 7860
```

Open http://localhost:7860, upload a recording, ask questions about it.

### REST API

```
GET  /api/echo/status              models loaded, load times, queue
POST /api/echo/jobs                multipart "file" -> 202 {"id"}
GET  /api/echo/jobs/{id}           poll: stage, percent, per-stage seconds, minutes, speakers + evidence
POST /api/echo/jobs/{id}/ask       {"question": "..."} -> {found, answer, cite: ["mm:ss"], passages}
```

### Deploy

One Docker image (`Dockerfile`, x86-64 or ARM64) for any host with about 6 GB of RAM. It bakes in
the open models; pyannote downloads at start-up with the `HF_TOKEN` secret.

- **Oracle Cloud Always Free** (Ampere A1, 2 cores / 12 GB, no cost): ECHO + Caddy for HTTPS with
  one script. Step by step: [`deploy/oracle/README.md`](deploy/oracle/README.md).
- **Hugging Face Docker Space** (CPU Basic, needs a PRO plan for personal accounts): the Space card
  is [`deploy/huggingface/README.md`](deploy/huggingface/README.md).

### Tests and evaluation

```bash
.venv/Scripts/python -m pytest -q tests                  # alignment, verifier, minutes checks
AMI_SET=test .venv/Scripts/python eval/run.py            # ECHO on the held-out meetings
AMI_SET=test .venv/Scripts/python eval/score.py          # -> eval/results/scores_test.json
```

## Layout

```
echo/             pipeline: text.py, address.py, speakers.py, minutes.py, rag.py, llm.py, models.py, pipeline.py, api.py
static/           the web page
tests/            unit tests
scripts/          model download (used by the Docker build)
eval/             evaluation runners, scorer, results (see eval/README.md)
deploy/           Oracle Cloud (compose + Caddy + setup script) and Hugging Face Space card
docs/             the results site (GitHub Pages), built by scripts/build_site.py from eval/results
```

## Limits

- Whisper mistakes flow downstream ("Craig" heard as "Greg" stays "Greg").
- pyannote sometimes splits one person into two labels, which costs names.
- ROUGE measures word overlap with one annotator's wording, not correctness.
- AMI is English role-play around one product; other meetings will differ.
- Model licences: Whisper (MIT), pyannote (MIT, gated on Hugging Face), MiniLM (Apache-2.0),
  **Qwen2.5-3B (Qwen Research License, non-commercial)**. The LLM is a setting; swap it for commercial use.

## Data

AMI Meeting Corpus, CC BY 4.0 (University of Edinburgh). Audio is not stored in this repository.
