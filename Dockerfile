FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install model server, RunPod SDK, and brotli decoders (br responses)
RUN pip install --no-cache-dir \
    vllm \
    runpod \
    requests \
    brotli \
    brotlicffi \
    "aiohttp[speedups]"

# Unbuffered Python logs
ENV PYTHONUNBUFFERED=1

# Copy your handler
COPY handler.py /src/handler.py

# Set working directory
WORKDIR /src

CMD bash -lc "\
  set -euxo pipefail; \
  echo '[boot] starting vLLM'; \
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --disable-log-requests \
    --gpu-memory-utilization 0.9 \
    > /tmp/vllm.log 2>&1 & \
  VLLM_PID=$!; \
  echo "[boot] vLLM pid ${VLLM_PID}"; \
  tail -F /tmp/vllm.log & \
  echo '[boot] starting handler'; \
  exec python -u /src/handler.py"

