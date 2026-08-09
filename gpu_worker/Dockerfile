FROM nvidia/cuda:12.9.2-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 worker

WORKDIR /app
COPY gpu_worker/requirements.txt /app/gpu_worker/requirements.txt
RUN python3 -m pip install --break-system-packages -r /app/gpu_worker/requirements.txt
COPY regressionlab /app/regressionlab
COPY gpu_worker /app/gpu_worker
RUN chown -R worker:worker /app

USER worker
CMD ["python3", "-m", "gpu_worker.handler"]
