FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install vLLM
RUN pip install vllm runpod

# Copy your handler
COPY handler.py /src/handler.py
