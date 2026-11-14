"""Run the portfolio's live ECHO backend (Gemini) on the same AMI meetings.

It imports the site's own transcribe() from the portfolio backend, so this is
the exact prompt, schema and model fallback list the live demo uses. One change:
the per-answer time limit is raised from 4 to 15 minutes, because AMI meetings
run up to 40 minutes and the site's limit is sized for short uploads.

Run with the portfolio backend's Python, from anywhere:
    D:\\Projects\\portfolio\\backend\\.venv\\Scripts\\python run_gemini.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(r"D:\Projects\portfolio\backend")
os.chdir(BACKEND)  # settings read backend/.env relative to here
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).parent))

from app.echo import transcribe as echo  # noqa: E402

from ami import MEETINGS, audio_path  # noqa: E402

echo.ANSWER_TIMEOUT = 900.0
OUT = Path(__file__).parent / "results" / "gemini"
OUT.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    for m in MEETINGS:
        target = OUT / f"{m}.json"
        if target.exists():
            print(f"{m}: already done", flush=True)
            continue
        answered = {}
        t = time.perf_counter()
        try:
            minutes = await echo.transcribe(
                audio_path(m).read_bytes(), "audio/wav", on_model=lambda name: answered.setdefault("model", name)
            )
        except echo.TranscribeError as exc:
            print(f"{m}: FAILED {exc}", flush=True)
            continue
        seconds = time.perf_counter() - t
        data = minutes.model_dump() | {"seconds": seconds, "model": answered.get("model")}
        target.write_text(json.dumps(data, indent=1), encoding="utf8")
        print(f"{m}: {len(minutes.transcript)} turns via {answered.get('model')} in {seconds:.0f}s", flush=True)


asyncio.run(main())
