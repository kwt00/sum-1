FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install model server & RunPod
RUN pip install vllm runpod

# Copy your handler
COPY handler.py /src/handler.py

# Set working directory
WORKDIR /src
