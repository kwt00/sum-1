import os
import time
from typing import List, Optional

import requests
import runpod
from pydantic import BaseModel

VLLM_BASE = "http://127.0.0.1:8000"
VLLM_MODELS_ENDPOINT = f"{VLLM_BASE}/v1/models"
VLLM_COMPLETIONS_ENDPOINT = f"{VLLM_BASE}/v1/completions"
VLLM_HEALTH_TIMEOUT = float(os.environ.get("VLLM_HEALTH_TIMEOUT", 30))

_vllm_ready = False


class ChatMessage(BaseModel):
    role: str
    content: str


class JobInput(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    max_tokens: int = 256
    temperature: float = 0.0


def wait_for_vllm():
    """Poll the vLLM /v1/models endpoint until ready or timeout."""
    global _vllm_ready
    if _vllm_ready:
        return

    deadline = time.monotonic() + VLLM_HEALTH_TIMEOUT
    last_error = None

    while time.monotonic() < deadline:
        try:
            resp = requests.get(VLLM_MODELS_ENDPOINT, timeout=2)
            if resp.ok:
                _vllm_ready = True
                return
            last_error = f"status={resp.status_code} body={resp.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        time.sleep(1)

    raise RuntimeError(
        f"vLLM did not become ready within {VLLM_HEALTH_TIMEOUT} seconds. "
        f"Last error: {last_error}"
    )

def make_prompt_from_messages(msgs):
    # Simple Qwen chat format
    prompt = ""
    for m in msgs:
        prompt += f"<|im_start|>{m.role}\n{m.content}\n<|im_end|>\n"
    prompt += "<|im_start|>assistant"
    return prompt

def handler(job):
    job_input = JobInput(**job["input"])

    # Ensure vLLM is reachable before first request
    wait_for_vllm()

    # Determine prompt
    if job_input.messages:
        prompt = make_prompt_from_messages(job_input.messages)
    elif job_input.prompt:
        # If direct prompt
        prompt = f"<|im_start|>system\nYou are a helpful assistant. Provide only the final answer.\n<|im_end|>\n{job_input.prompt}"
    else:
        raise ValueError("Must provide prompt or messages")

    payload = {
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B"),
        "prompt": prompt,
        "max_tokens": job_input.max_tokens,
        "temperature": job_input.temperature,
        "stop": ["</think>", "<|im_end|>"]
    }

    r = requests.post(VLLM_COMPLETIONS_ENDPOINT, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()

runpod.serverless.start({"handler": handler})
