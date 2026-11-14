---
title: ECHO
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Meeting minutes on open models, every claim checked
---

# ECHO

Meeting recording in; who said what, decisions, action items and cited answers out.
Open models on 2 CPU cores; no paid API.

| Step | What runs |
|---|---|
| Transcribe | faster-whisper `small` (int8) with a voice-activity filter; a timestamp on every word |
| Who spoke | `pyannote/speaker-diarization-3.1`, 2 to 8 speakers |
| Align | each word goes to the speaker turn it overlaps most; the transcript is cut where the speaker changes |
| Speakers | Qwen2.5-3B-Instruct (llama.cpp) proposes name, role and a quote; code keeps only what the transcript backs up, plus roll-call answers and addressee voting |
| Minutes | the same model, section by section: decisions, action items (owner, due), open questions, each with its `[mm:ss]`; code checks each against its line |
| Q&A | LangChain: BM25 + FAISS (all-MiniLM-L6-v2) fused; answers cite the times they come from |

Recordings longer than `ECHO_MAX_SECONDS` (default 300) are trimmed. One job runs at a time.

## REST API

```
GET  /api/echo/status              models loaded, load times, queue
POST /api/echo/jobs                multipart "file" -> 202 {"id"}
GET  /api/echo/jobs/{id}           poll: stage, percent, stage_seconds, minutes, speakers + evidence
POST /api/echo/jobs/{id}/ask       {"question": "..."} -> found, answer, cite, passages
```

## Secrets

`HF_TOKEN`: a token whose account has accepted the terms of `pyannote/speaker-diarization-3.1` and
`pyannote/segmentation-3.0`. Without it the diarization model can't load and `/api/echo/status` shows why.

## Hosting

Needs about 6 GB of RAM. On Hugging Face this is a Docker Space on CPU Basic (2 vCPU, 16 GB; no
hourly cost, but Docker Spaces need a PRO plan for personal accounts). Any VM with Docker works too.

## Publishing

A Space's repository is this project's root with this file as its `README.md` (the front matter
above is the Space config). Docker Spaces need a PRO plan for personal accounts.
