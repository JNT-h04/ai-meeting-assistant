"""Load each model once, on first use, and keep it. All CPU unless ECHO_DEVICE=cuda.

The target host (Hugging Face CPU Basic) has 2 vCPUs and 16 GB of RAM. Together the
models need about 4 GB: faster-whisper small int8 (~0.5), pyannote (~0.3),
MiniLM (~0.1) and Qwen2.5-3B 4-bit (~2.5 with an 8k context).
"""

import os
import threading
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # noqa: E402

THREADS = int(os.environ.get("ECHO_THREADS", "2"))
DEVICE = os.environ.get("ECHO_DEVICE", "cpu")
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)

ASR_MODEL = os.environ.get("ECHO_ASR_MODEL", "small")  # faster-whisper name or CTranslate2 repo
DIAR_MODEL = "pyannote/speaker-diarization-3.1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()
_cache: dict[str, object] = {}
load_seconds: dict[str, float] = {}
load_errors: dict[str, str] = {}


def _once(name: str, build):
    with _guard:
        lock = _locks.setdefault(name, threading.Lock())
    with lock:  # one lock per model, so warming the LLM doesn't block a job that needs Whisper
        if name not in _cache:
            t = time.perf_counter()
            _cache[name] = build()
            load_errors.pop(name, None)
            load_seconds[name] = round(time.perf_counter() - t, 1)
        return _cache[name]


def loaded() -> list[str]:
    return sorted(_cache)


def asr():
    def build():
        from faster_whisper import WhisperModel

        return WhisperModel(ASR_MODEL, device=DEVICE, compute_type="int8" if DEVICE == "cpu" else "float16",
                            cpu_threads=THREADS)

    return _once("asr", build)


def _patch_pyannote() -> None:
    import torchaudio

    # pyannote.audio 3.x names two torchaudio things that newer releases removed.
    # Audio reaches it already decoded, so they only need to exist.
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = type("AudioMetaData", (), {})
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]


def diarizer():
    def build():
        _patch_pyannote()
        from pyannote.audio import Pipeline

        # Gated model: the token's account must have accepted the terms of
        # speaker-diarization-3.1 and segmentation-3.0. On the Space it is a secret.
        pipe = Pipeline.from_pretrained(DIAR_MODEL, use_auth_token=os.environ.get("HF_TOKEN") or None)
        if pipe is None:
            raise RuntimeError("pyannote refused the HF_TOKEN: accept the model terms, then set a valid token")
        pipe.to(torch.device(DEVICE))
        return pipe

    return _once("diarizer", build)


def embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return _once("embeddings", lambda: HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        # Unit vectors, so FAISS inner product = cosine similarity.
        encode_kwargs={"normalize_embeddings": True},
    ))


def language_model():
    from echo import llm

    return _once("llm", llm.model)


def warm_up() -> None:
    """Load everything in the background at start-up so the first visitor doesn't wait for it."""
    for name, build in (("asr", asr), ("diarizer", diarizer), ("llm", language_model), ("embeddings", embeddings)):
        try:
            build()
        except Exception as exc:  # shown by /api/echo/status; a job retries the load
            load_errors[name] = f"{exc.__class__.__name__}: {str(exc)[:200]}"
