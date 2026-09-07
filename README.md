# AI Meeting Assistant

Upload a meeting recording and get back a speaker-attributed transcript and a summary. Combines Whisper (ASR), pyannote (speaker diarization), and BART (summarization) behind a Streamlit UI.

## Architecture
See [`architecture/`](architecture/) for diagrams:
- [Processing pipeline](architecture/pipeline.md) — audio → transcript → diarization → summary
- [Speaker attribution](architecture/speaker_attribution.md) — how anonymous speaker IDs get mapped to real names

## Repository Structure
```
ai-meeting-assistant/
├── app/
│   ├── app.py             # Streamlit UI
│   └── meeting_model.py    # ASR + diarization + summarization pipeline
├── notebooks/               # Prototyping notebooks
├── samples/                 # Sample AMI Corpus audio + example outputs (see samples/README.md)
├── architecture/             # System diagrams
├── .env.example
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup
```bash
git clone <this-repo-url>
cd ai-meeting-assistant

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Get a Hugging Face access token, accept the terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), then:
```bash
copy .env.example .env    # Windows: copy, macOS/Linux: cp
# edit .env and set HF_TOKEN=<your token>
```

## Running the App
```bash
cd app
streamlit run app.py
```
Upload a `.wav`/`.mp3` recording to get a diarized transcript and an automatic summary.

## How It Works
1. **Transcription** — `openai/whisper-small` produces timestamped text segments.
2. **Diarization** — `pyannote/speaker-diarization-3.1` identifies who spoke when (2–8 speakers).
3. **Speaker attribution** — each transcript segment is assigned to the speaker with the largest time overlap; if someone says "my name is X" / "this is X", that name replaces their anonymous speaker ID for the rest of the transcript.
4. **Summarization** — the merged transcript is chunked (~200 words) and summarized with `facebook/bart-large-cnn`.

Full diagram: [`architecture/pipeline.md`](architecture/pipeline.md).

## Sample Data
Six AMI Meeting Corpus recordings were used for development — see [`samples/README.md`](samples/README.md) for how to obtain them (not committed to this repo; ~350 MB).

## Security Note
Earlier revisions of this project had a Hugging Face access token hardcoded directly in `meeting_model.py` and in two prototyping notebooks. This has been fixed — the token is now read from the `HF_TOKEN` environment variable (see `.env.example`). **If those old tokens were ever pushed to GitHub, treat them as compromised and revoke/rotate them at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)** — removing them from new commits does not remove them from git history.

## License
[MIT](LICENSE)

## Acknowledgements
- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) — sample audio
- [OpenAI Whisper](https://github.com/openai/whisper), [pyannote.audio](https://github.com/pyannote/pyannote-audio), [BART](https://huggingface.co/facebook/bart-large-cnn)
