"""ECHO replica REST API. Same contract as the portfolio's ECHO, so its page can point here.

    GET  /api/echo/status          models loaded, queue length, limits
    POST /api/echo/jobs            upload a recording -> 202 {id}
    GET  /api/echo/jobs/{id}       poll until stage is "done" or "error"
    POST /api/echo/jobs/{id}/ask   {"question": "..."} -> answer from the transcript (RAG)

One job runs at a time: the Space has 2 vCPUs, and two pipelines at once would
just take twice as long each. Everything is in memory; nothing is written to disk.
"""

import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from echo import llm, models, pipeline

MAX_BYTES = 50 * 1024 * 1024
MAX_SECONDS = float(os.environ.get("ECHO_MAX_SECONDS", "300"))
JOB_TTL = 60 * 60
MAX_JOBS = 30
MAX_QUEUE = 5
UPLOADS_PER_HOUR = int(os.environ.get("ECHO_UPLOADS_PER_HOUR", "5"))
QUESTIONS_PER_HOUR = int(os.environ.get("ECHO_QUESTIONS_PER_HOUR", "40"))

Stage = Literal["queued", "decoding", "transcribing", "diarizing", "aligning", "identifying",
                "writing minutes", "indexing", "done", "error"]
PERCENT = {"queued": 2, "decoding": 4, "transcribing": 8, "diarizing": 25, "aligning": 55,
           "identifying": 58, "writing minutes": 72, "indexing": 96, "done": 100}


class Turn(BaseModel):
    speaker: str
    label: str
    start: str
    start_s: float
    end_s: float
    text: str


class Action(BaseModel):
    task: str
    owner: str
    due: str


class Minutes(BaseModel):
    title: str
    summary: str
    topics: list[str]
    decisions: list[str]
    actions: list[Action]
    open_questions: list[str]
    method: str
    sections: int
    decisions_at: list[str] = Field(default_factory=list)  # [mm:ss] where each decision was said
    actions_at: list[str] = Field(default_factory=list)
    dropped_unverified: int = 0  # items the model proposed that the transcript didn't back up
    transcript: list[Turn]


class Speaker(BaseModel):
    label: str
    display: str
    name: str
    role: str
    evidence: str
    verified: bool
    rejected: list[str]
    talk_seconds: float


class Job(BaseModel):
    id: str
    filename: str
    stage: Stage = "queued"
    percent: int = 2
    queue_position: int = 0
    seconds: float = 0.0
    model: str = ""
    audio_bytes: int = 0
    audio_seconds: float = 0.0
    trimmed_from: float | None = None
    speaker_turns: int = 0
    word_count: int = 0
    speakers: list[Speaker] = Field(default_factory=list)
    llm: dict = Field(default_factory=dict)
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    error: str = ""
    minutes: Minutes | None = None
    created: float = 0.0


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=300)


class Answer(BaseModel):
    question: str
    found: bool
    answer: str
    cite: list[str]
    passages: list[str]
    seconds: float
    model: str


_jobs: dict[str, Job] = {}
_indexes: dict[str, object] = {}
_work: "queue.Queue[tuple[Job, bytes]]" = queue.Queue()
_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_hits_lock = threading.Lock()
MODEL_LINE = f"faster-whisper {models.ASR_MODEL} + {models.DIAR_MODEL} + {llm.name()}"


def _client(request: Request) -> str:
    # Behind the Space's proxy the visitor is the first address in X-Forwarded-For.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


def _limit(request: Request, kind: str, per_hour: int) -> None:
    now = time.time()
    with _hits_lock:
        hits = _hits[(kind, _client(request))]
        while hits and now - hits[0] > 3600:
            hits.popleft()
        if len(hits) >= per_hour:
            wait = int(3600 - (now - hits[0])) // 60 + 1
            raise HTTPException(429, f"limit reached ({per_hour} an hour). try again in {wait} min")
        hits.append(now)


def _forget_old() -> None:
    now = time.time()
    for key in [k for k, j in _jobs.items() if now - j.created > JOB_TTL and j.stage in ("done", "error")]:
        _jobs.pop(key, None)
        _indexes.pop(key, None)
    while len(_jobs) >= MAX_JOBS:
        old = next(iter(_jobs))
        _jobs.pop(old, None)
        _indexes.pop(old, None)


def _worker() -> None:
    while True:
        job, data = _work.get()
        started = time.time()

        def on_stage(stage: str) -> None:
            job.stage = stage  # type: ignore[assignment]
            job.percent = PERCENT.get(stage, job.percent)

        try:
            result = pipeline.run(data, MAX_SECONDS, on_stage)
            job.minutes = Minutes(**result["minutes"])
            job.audio_seconds = result["audio_seconds"]
            job.trimmed_from = result["trimmed_from"]
            job.speaker_turns = result["speaker_turns"]
            job.word_count = result["word_count"]
            job.speakers = [Speaker(**s) for s in result["speakers"]]
            job.llm = result["llm"]
            job.stage_seconds = result["stage_seconds"]
            if result["index"] is not None:
                _indexes[job.id] = result["index"]
            job.stage, job.percent = "done", 100
        except pipeline.PipelineError as exc:
            job.stage, job.error = "error", str(exc)
        except Exception as exc:  # never hand a stack trace to a visitor
            job.stage, job.error = "error", f"unexpected failure ({exc.__class__.__name__}: {str(exc)[:160]})"
        finally:
            job.seconds = round(time.time() - started, 1)
            del data
            _work.task_done()


app = FastAPI(title="ECHO replica", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get(
        "ECHO_ORIGINS", "https://jnt-h04.github.io,http://localhost:5173").split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _start() -> None:
    threading.Thread(target=_worker, name="echo-worker", daemon=True).start()
    if os.environ.get("ECHO_WARM", "1") == "1":
        threading.Thread(target=models.warm_up, name="echo-warm", daemon=True).start()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/echo/status")
def status() -> dict:
    needed = {"asr", "diarizer", "llm", "embeddings"}
    return {
        "ready": needed <= set(models.loaded()),
        "loaded": models.loaded(),
        "load_seconds": models.load_seconds,
        "load_errors": models.load_errors,
        "queue": _work.qsize(),
        "max_seconds": MAX_SECONDS,
        "threads": models.THREADS,
        "models": {"asr": f"faster-whisper {models.ASR_MODEL} int8", "diarization": models.DIAR_MODEL,
                   "llm": llm.name(), "embeddings": models.EMBED_MODEL},
    }


@app.post("/api/echo/jobs", status_code=202)
async def submit(request: Request, file: UploadFile) -> dict:
    if _work.qsize() >= MAX_QUEUE:
        raise HTTPException(503, "busy: too many recordings waiting. try again in a few minutes")
    _limit(request, "upload", UPLOADS_PER_HOUR)
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "recording too large (max 50 MB)")
    if not data:
        raise HTTPException(422, "that file is empty")
    _forget_old()
    job = Job(id=uuid.uuid4().hex, filename=file.filename or "recording", created=time.time(),
              audio_bytes=len(data), model=MODEL_LINE)
    _jobs[job.id] = job
    _work.put((job, data))
    return {"id": job.id}


@app.get("/api/echo/jobs/{job_id}", response_model=Job)
def job_status(job_id: str) -> Job:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found or expired. upload the recording again")
    if job.stage == "queued":
        waiting = [j.id for j, _ in list(_work.queue)]
        job.queue_position = waiting.index(job.id) + 1 if job.id in waiting else 0
    return job


@app.post("/api/echo/jobs/{job_id}/ask", response_model=Answer)
def ask(job_id: str, body: Question, request: Request) -> Answer:
    # A plain def: FastAPI runs it in a thread, so a slow answer doesn't block polling.
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found or expired. upload the recording again")
    if job.stage != "done":
        raise HTTPException(409, "the minutes aren't ready yet")
    index = _indexes.get(job_id)
    if index is None:
        raise HTTPException(422, "nothing was said that could be searched")
    _limit(request, "ask", QUESTIONS_PER_HOUR)
    return Answer(**index.ask(body.question))


STATIC = Path(__file__).resolve().parent.parent / "static"


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(STATIC / "index.html")
