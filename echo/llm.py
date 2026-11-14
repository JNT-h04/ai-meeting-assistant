"""The local language model: Qwen2.5-3B-Instruct, 4-bit GGUF, run by llama.cpp on CPU.

Every call asks for JSON that matches a schema. llama.cpp turns the schema into a
grammar, so the model *cannot* produce anything else: no parsing failures, no
"Sure! Here is your JSON:" preambles.
"""

import json
import os
import threading
import time
from pathlib import Path

REPO = os.environ.get("ECHO_LLM_REPO", "Qwen/Qwen2.5-3B-Instruct-GGUF")
FILE = os.environ.get("ECHO_LLM_FILE", "qwen2.5-3b-instruct-q4_k_m.gguf")
# A GGUF already on disk (ECHO_LLM_PATH, else .models/ next to the package); otherwise the HF cache.
LOCAL = Path(os.environ.get("ECHO_LLM_PATH") or Path(__file__).resolve().parent.parent / ".models" / FILE)
CTX = int(os.environ.get("ECHO_LLM_CTX", "8192"))
THREADS = int(os.environ.get("ECHO_THREADS", "2"))

_lock = threading.Lock()  # llama.cpp contexts are not thread-safe: one call at a time
_model = None
stats = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0}


def model():
    global _model
    if _model is None:
        from llama_cpp import Llama

        path = LOCAL if LOCAL.exists() else None
        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(REPO, FILE)
        _model = Llama(model_path=str(path), n_ctx=CTX, n_threads=THREADS, n_batch=256, verbose=False)
    return _model


def name() -> str:
    return FILE.removesuffix(".gguf")


def tokens(text: str) -> int:
    return len(model().tokenize(text.encode("utf8"), add_bos=False))


def json_call(system: str, user: str, schema: dict, max_tokens: int = 700) -> dict:
    with _lock:
        t = time.perf_counter()
        out = model().create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object", "schema": schema},
            temperature=0.1,
            repeat_penalty=1.1,
            max_tokens=max_tokens,
        )
        usage = out["usage"]
        stats["calls"] += 1
        stats["prompt_tokens"] += usage["prompt_tokens"]
        stats["output_tokens"] += usage["completion_tokens"]
        stats["seconds"] = round(stats["seconds"] + time.perf_counter() - t, 1)
    text = out["choices"][0]["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Only possible if max_tokens cut the JSON off mid-way.
        raise RuntimeError(f"the language model ran out of room ({usage['completion_tokens']} tokens)") from None


def text_call(system: str, user: str, max_tokens: int = 300) -> str:
    with _lock:
        out = model().create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1, repeat_penalty=1.1, max_tokens=max_tokens,
        )
    return out["choices"][0]["message"]["content"].strip()
