FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    safetensors \
    sentencepiece \
    bitsandbytes

WORKDIR /workspace

COPY model.py train.py /workspace/

ENTRYPOINT ["python3", "train.py"]
