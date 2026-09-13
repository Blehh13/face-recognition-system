# Face Recognition Identification System
#
# Builds dlib from source, which is the slow part (~10 minutes) and the reason
# most free hosts with small build containers fail. Hugging Face Spaces has
# enough memory; Render's 512 MB free tier does not.
#
#   docker build -t facerec .
#   docker run -p 7860:7860 facerec
#
# Configuration (see app.py):
#   FACEREC_HOST    interface to bind
#   FACEREC_PORT    port
#   FACEREC_SECRET  Flask secret key
#   FACEREC_DB      database path — use a .sqlite extension for the
#                   multi-process-safe backend
#   FACEREC_DEMO    "1" wipes the database at start-up (see app.py)

# ---------------------------------------------------------------- frontend
# Built inside the image so it always matches the committed source. Relying on
# a locally-built static/dist made the image depend on whoever last ran
# `npm run build`, and a clean clone would have shipped no frontend at all.
FROM node:20-slim AS frontend

WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build
# vite writes to ../static/dist; hoist it somewhere predictable to copy from.
RUN mkdir -p /out && cp -r /static/dist /out/dist

# ------------------------------------------------------------ python build
FROM python:3.12-slim AS build

# dlib needs a compiler and CMake. No GL/GLib here: the runtime uses the
# headless OpenCV wheel, which does not link against them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Runtime dependencies only — see requirements-app.txt. Installing the full
# development set pulled scipy, pandas, scikit-learn and matplotlib into the
# image, none of which the server imports.
COPY requirements-app.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-app.txt

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim

# dlib links against OpenBLAS at runtime. Headless OpenCV needs nothing else.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Hugging Face Spaces runs as a non-root user and expects port 7860.
RUN useradd -m -u 1000 appuser

WORKDIR /app
# --chown on the COPY itself. A later `chown -R` rewrites every file and so
# duplicates the whole application layer.
COPY --chown=appuser:appuser . .
COPY --from=frontend --chown=appuser:appuser /out/dist ./static/dist
RUN chmod +x /app/docker-entrypoint.sh
RUN mkdir -p /app/database /app/models \
    && chown appuser:appuser /app/database /app/models
USER appuser

ENV FACEREC_HOST=0.0.0.0 \
    FACEREC_PORT=7860 \
    FACEREC_DB=/app/database/enrolled_faces.sqlite \
    FACEREC_DEMO=1 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/health')"

# Two workers, which is precisely why the store had to stop being a JSON file:
# the old one silently lost an enrolment whenever a second worker wrote.
# --timeout 120 covers the first request, which loads the dlib models.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
