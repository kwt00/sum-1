import runpod
from pydantic import BaseModel
import requests
import os

VLLM_ENDPOINT = "http://127.0.0.1:8000/v1/completions"

class JobInput(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0

def handler(job):
    job_input = JobInput(**job["input"])

    payload = {
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B"),
        "prompt": job_input.prompt,
        "max_tokens": job_input.max_tokens,
        "temperature": job_input.temperature,
        "stop": ["<|im_end|>", "</think>"]
    }

    resp = requests.post(VLLM_ENDPOINT, json=payload, timeout=300)
    resp.raise_for_status()

    return resp.json()

runpod.serverless.start({"handler": handler})
