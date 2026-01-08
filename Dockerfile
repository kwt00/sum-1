FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install model server & RunPod
RUN pip install --no-cache-dir vllm runpod requests

# Copy your handler
COPY handler.py /src/handler.py

# Set working directory
WORKDIR /src

CMD bash -lc "\
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --disable-log-requests \
    --gpu-memory-utilization 0.9 \
    > /tmp/vllm.log 2>&1 & \
  exec python -u /src/handler.py"
