"""Bake the ungated models into the image, so a Space waking from sleep doesn't download ~3 GB first.

pyannote is gated, so it downloads at start-up with the HF_TOKEN secret.
"""
import os

from huggingface_hub import hf_hub_download, snapshot_download

# faster-whisper "small" = this CTranslate2 conversion
snapshot_download("Systran/faster-whisper-small")
snapshot_download("sentence-transformers/all-MiniLM-L6-v2",
                  ignore_patterns=["*.bin", "*.h5", "*.ot", "onnx/*", "openvino/*", "*.msgpack"])
hf_hub_download(os.environ.get("ECHO_LLM_REPO", "Qwen/Qwen2.5-3B-Instruct-GGUF"),
                os.environ.get("ECHO_LLM_FILE", "qwen2.5-3b-instruct-q4_k_m.gguf"))
print("models baked in", flush=True)
