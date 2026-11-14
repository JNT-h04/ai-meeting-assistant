"""Run ECHO on the AMI meetings and save everything score.py needs.

    .venv/Scripts/python eval/run.py            every meeting of the split, skipping finished ones
    .venv/Scripts/python eval/run.py ES2002a    one meeting

AMI_SET=dev (default) or test picks the split (see ami.py). Transcripts and diarization are
saved separately (results/words, results/diarize) and reused, so a change to the
language-model steps can be re-measured without redoing the audio work.

ECHO_THREADS (default 4 here) sets the CPU threads. On a 16 GB laptop run one meeting per
process: several at once ran the machine out of memory.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ECHO_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import soundfile as sf  # noqa: E402

from ami import MEETINGS, audio_path  # noqa: E402
from echo import llm, pipeline  # noqa: E402

RESULTS = HERE / "results"
OUT, WORDS, DIARIZE = RESULTS / "echo", RESULTS / "words", RESULTS / "diarize"
for d in (OUT, WORDS, DIARIZE):
    d.mkdir(parents=True, exist_ok=True)


def load_audio(meeting: str):
    audio, sr = sf.read(audio_path(meeting), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != pipeline.RATE:
        import torch
        import torchaudio.functional as F

        audio = F.resample(torch.from_numpy(audio), sr, pipeline.RATE).numpy()
    return audio


def run(meeting: str) -> None:
    out = OUT / f"{meeting}.json"
    if out.exists():
        print(f"{meeting}: already done, skipped", flush=True)
        return
    words_file, dia_file = WORDS / f"{meeting}.json", DIARIZE / f"{meeting}.json"
    # Only touch the audio when something still has to be computed from it.
    audio = None if words_file.exists() and dia_file.exists() else load_audio(meeting)
    if audio is not None:
        minutes_long = len(audio) / pipeline.RATE / 60
    else:
        minutes_long = sf.info(str(audio_path(meeting))).duration / 60
    print(f"{meeting}: {minutes_long:.1f} min of audio", flush=True)
    timings: dict[str, float] = {}

    if words_file.exists():
        saved = json.loads(words_file.read_text(encoding="utf8"))
        words, timings["transcribing"] = saved["words"], saved["seconds"]
    else:
        t = time.perf_counter()
        words = pipeline.transcribe(audio)
        timings["transcribing"] = round(time.perf_counter() - t, 1)
        words_file.write_text(json.dumps({"words": words, "seconds": timings["transcribing"]}), encoding="utf8")
    print(f"  transcribed: {len(words)} words in {timings['transcribing']}s", flush=True)

    if dia_file.exists():
        saved = json.loads(dia_file.read_text(encoding="utf8"))
        turns, timings["diarizing"] = saved["turns"], saved.get("seconds", 0.0)
    else:
        t = time.perf_counter()
        turns = pipeline.diarize(audio)
        timings["diarizing"] = round(time.perf_counter() - t, 1)
        dia_file.write_text(json.dumps({"turns": turns, "seconds": timings["diarizing"], "device": "cpu",
                                        "threads": int(os.environ["ECHO_THREADS"])}, indent=1), encoding="utf8")
    print(f"  diarized: {len(turns)} turns", flush=True)

    before = dict(llm.stats)
    result = pipeline.understand(words, turns, lambda s: print(f"  {s}", flush=True), timings)
    result.pop("index")
    result.update(
        meeting=meeting, audio_minutes=round(minutes_long, 2), stage_seconds=timings,
        threads=int(os.environ["ECHO_THREADS"]),
        llm={"model": llm.name(), "calls": llm.stats["calls"] - before["calls"],
             "prompt_tokens": llm.stats["prompt_tokens"] - before["prompt_tokens"],
             "output_tokens": llm.stats["output_tokens"] - before["output_tokens"],
             "seconds": round(llm.stats["seconds"] - before["seconds"], 1)},
    )
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf8")
    m = result["minutes"]
    print(f"  done: {[s['display'] for s in result['speakers']]}, {len(m['decisions'])} decisions, "
          f"{len(m['actions'])} actions, {m['method']} ({m['sections']} sections), {timings}", flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:] or MEETINGS:
        run(m)
