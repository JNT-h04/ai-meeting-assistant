# ECHO as one container: the API and the web page on port 7860.
# Needs about 6 GB of RAM. Builds on x86-64 and on ARM64 (e.g. Oracle Cloud's Ampere A1).
FROM python:3.11-slim

# build-essential + cmake: only used on ARM64, where llama.cpp is compiled for this CPU (see below).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as uid 1000; elsewhere it's simply a non-root user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
# x86-64: llama-cpp-python's prebuilt CPU wheel. ARM64: compiled here (~10 min) so llama.cpp uses
# exactly this CPU's instructions; a prebuilt ARM wheel can stop with "illegal instruction".
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        pip install --no-cache-dir --user --no-binary llama-cpp-python -r requirements.txt; \
    else \
        pip install --no-cache-dir --user -r requirements.txt \
            --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu; \
    fi

# Bake the ungated models (~3 GB) into the image, so a restart doesn't download them again.
# pyannote is gated: it downloads at start-up with HF_TOKEN.
COPY --chown=user scripts/fetch_models.py scripts/
RUN python scripts/fetch_models.py

COPY --chown=user echo echo
COPY --chown=user static static

ENV ECHO_THREADS=2
EXPOSE 7860
CMD ["uvicorn", "echo.api:app", "--host", "0.0.0.0", "--port", "7860"]
