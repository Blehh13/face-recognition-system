# Face Recognition Identification System
#
# Builds in about a minute and needs no compiler: the default engine is
# YuNet + SFace through opencv-python's DNN module. An earlier version of this
# file compiled dlib from source, which took ten minutes and exhausted the
# memory of most small build containers.
#
#   docker build -t facerec .
#   docker run -p 7860:7860 facerec
#
# Configuration (see app.py):
#   FACEREC_HOST    interface to bind
#   FACEREC_PORT    port
#   FACEREC_SECRET  Flask secret key
#   FACEREC_DB      database path — a .sqlite extension selects the
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

WORKDIR /app

# Runtime dependencies only — see requirements-app.txt. The full development
# set pulls scipy, pandas, scikit-learn and matplotlib, none of which the
# server imports.
COPY requirements-app.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-app.txt

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim

# No apt packages: the headless OpenCV wheel is self-contained, and with the
# default engine there is no dlib to link against.
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Hugging Face Spaces runs as a non-root user and expects port 7860.
RUN useradd -m -u 1000 appuser

WORKDIR /app
# --chown on the COPY itself. A later `chown -R` rewrites every file and so
# duplicates the whole application layer.
COPY --chown=appuser:appuser . .
COPY --from=frontend --chown=appuser:appuser /out/dist ./static/dist
RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /app/database /app/models \
    && chown appuser:appuser /app/database /app/models
USER appuser

ENV FACEREC_HOST=0.0.0.0 \
    FACEREC_PORT=7860 \
    FACEREC_DB=/app/database/enrolled_faces.sqlite \
    FACEREC_DEMO=1 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/health')"

# Two workers, which is why the store had to stop being a JSON file: the old
# one silently lost an enrolment whenever a second worker wrote.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
