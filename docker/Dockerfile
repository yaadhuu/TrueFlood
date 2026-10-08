# Docker is OPTIONAL for this project. The deploy path is plain Python on
# Render/Railway (see Procfile / render.yaml). This image exists because some
# reviewers look for a Dockerfile, and because it is a convenient way to run
# the drift monitor on a schedule.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_PATH=/data/flood_data.db

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY ml_pipeline/ ./ml_pipeline/
COPY monitoring/ ./monitoring/
COPY tests/ ./tests/

RUN mkdir -p /data && \
    useradd --create-home --uid 10001 floodsense && \
    chown -R floodsense /data /app
USER floodsense

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4).status==200 else 1)"

# --workers 1 is load-bearing; see app.py's WEB_CONCURRENCY assertion.
CMD ["gunicorn", "--chdir", "backend", "app:app", \
     "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]
