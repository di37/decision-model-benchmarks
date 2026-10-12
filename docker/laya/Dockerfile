FROM python:3.11-slim-bookworm

ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX=cu128

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_DISABLE_NATIVE_JIT=1

RUN python -m pip install \
      "torch==${TORCH_VERSION}" \
      --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
    && python -m pip install "laya[serve]==0.3.21"

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
