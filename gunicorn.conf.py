import os


bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "2"))
timeout = int(os.getenv("GUNICORN_TIMEOUT_SECONDS", "180"))
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
capture_output = True

# SQLite, filesystem artifacts, and in-memory rate limits intentionally use a
# single process for the hobby deployment. Move these services out of process
# before increasing WEB_CONCURRENCY.
if workers != 1:
    raise RuntimeError(
        "RegressAI currently requires WEB_CONCURRENCY=1."
    )
