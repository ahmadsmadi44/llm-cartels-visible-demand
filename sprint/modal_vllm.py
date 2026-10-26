"""[sprint tooling] OpenAI-compatible vLLM server on Modal for the agents (buyers, sellers, overseer).

Setup:    modal secret create collusion-vllm-key VLLM_API_KEY=<random>
Deploy:   SERVE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 modal deploy sprint/modal_vllm.py
Endpoint: https://<workspace>--collusion-vllm-serve.modal.run/v1   (served model name = $SERVED_NAME)
Stop:     modal app stop collusion-vllm
Scales to zero after 10 idle minutes, so an idle deployment costs nothing.
"""
import os
import subprocess

import modal

MODEL = os.environ.get("SERVE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
SERVED_NAME = os.environ.get("SERVED_NAME", "qwen3-30b")
GPU = os.environ.get("SERVE_GPU", "H100")
MAX_LEN = os.environ.get("SERVE_MAX_LEN", "16384")
APP_NAME = os.environ.get("SERVE_APP", "collusion-vllm")

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_SERVE_MODEL": MODEL, "VLLM_SERVED_NAME": SERVED_NAME,
          "VLLM_MAX_LEN": MAX_LEN})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * 3600,
    scaledown_window=120,
    max_containers=1,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("collusion-vllm-key")],  # holds VLLM_API_KEY
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=8000, startup_timeout=1800)
def serve():
    cmd = [
        "vllm", "serve", os.environ["VLLM_SERVE_MODEL"],
        "--served-model-name", os.environ["VLLM_SERVED_NAME"],
        "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", os.environ["VLLM_MAX_LEN"],
        "--gpu-memory-utilization", "0.92",
        "--max-num-seqs", "160",
        "--enable-prefix-caching",
    ]
    if os.environ.get("VLLM_API_KEY"):
        cmd += ["--api-key", os.environ["VLLM_API_KEY"]]
    subprocess.Popen(cmd)
