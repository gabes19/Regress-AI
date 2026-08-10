FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_ROOT=/data \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-fonts-recommended \
        gosu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.lock.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.lock.txt

COPY app.py config.py gunicorn.conf.py docker-entrypoint.sh ./
COPY regressionlab ./regressionlab
COPY templates ./templates
COPY sample_data ./sample_data

RUN groupadd --system regressai \
    && useradd --system --gid regressai --home-dir /app regressai \
    && mkdir -p /data/uploads /data/reports /data/instance \
    && chown -R regressai:regressai /app /data \
    && chmod 755 /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
