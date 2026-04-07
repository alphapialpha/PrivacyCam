# =============================================================================
# PrivacyCam Dockerfile
# =============================================================================
FROM python:3.13-slim

# ---------------------------------------------------------------------------
# System dependencies
#   libgl1 + libglib2.0-0  — required by OpenCV headless at import time
#   curl + ca-certificates — used to download supercronic
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# supercronic — cron daemon designed for containers (logs to stdout, no PID issues)
# https://github.com/aptible/supercronic
#
# Pinned to v0.2.44. SHA1 checksums from the official releases page:
#   https://github.com/aptible/supercronic/releases/tag/v0.2.44
# The fork-exec "no such file" bug present in v0.2.33 was fixed in v0.2.36.
# ---------------------------------------------------------------------------
ARG SUPERCRONIC_VERSION=0.2.44
RUN ARCH="$(uname -m)" && \
    case "${ARCH}" in \
      x86_64)  SC_ARCH=amd64; SC_SHA1=6eb0a8e1e6673675dc67668c1a9b6409f79c37bc ;; \
      aarch64) SC_ARCH=arm64; SC_SHA1=6c6cba4cde1dd4a1dd1e7fb23498cde1b57c226c ;; \
      *)        echo "Unsupported architecture: ${ARCH}" >&2 && exit 1 ;; \
    esac && \
    SC_BIN="supercronic-linux-${SC_ARCH}" && \
    curl -fsSLO "https://github.com/aptible/supercronic/releases/download/v${SUPERCRONIC_VERSION}/${SC_BIN}" && \
    echo "${SC_SHA1}  ${SC_BIN}" | sha1sum -c - && \
    chmod +x "${SC_BIN}" && \
    mv "${SC_BIN}" "/usr/local/bin/${SC_BIN}" && \
    ln -s "/usr/local/bin/${SC_BIN}" /usr/local/bin/supercronic

# ---------------------------------------------------------------------------
# Python dependencies
#
# opencv-python-headless: headless build — no X11/display dependency.
#   Using the GUI build (opencv-python) in a headless container causes
#   ImportError at runtime.
#
# ultralytics: provides solutions.ObjectBlurrer and the YOLO model runner.
# paramiko:    pure-Python SSH/SFTP client; replaces rclone entirely.
#
# For production deployments, pin exact versions, e.g.:
#   ultralytics==8.4.27  opencv-python-headless==4.11.0.86  paramiko==3.5.1
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
      ultralytics \
      opencv-python-headless \
      paramiko

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------
WORKDIR /app

# Silence Ultralytics telemetry and verbose banners at build and runtime.
# YOLO_CONFIG_DIR redirects the settings/cache directory into /app so that
# the chown below covers it and the non-root user can write to it at runtime.
ENV YOLO_VERBOSE=False \
    YOLO_TELEMETRY=False \
    YOLO_CONFIG_DIR=/app/ultralytics_cache

# Pre-download the YOLO model into the image so the container starts offline.
# Working directory is set to /app/models so the model file is written there
# (Ultralytics downloads relative-path model names to the current directory).
# Change yolov8n.pt to a different model name here AND in .env MODEL_PATH if
# you want higher accuracy (e.g. yolov8s.pt) at the cost of larger image size.
RUN mkdir -p /app/models
WORKDIR /app/models
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
WORKDIR /app

# Copy application scripts
COPY privacy_cam.py entrypoint.sh healthcheck.sh ./
RUN chmod +x entrypoint.sh healthcheck.sh

# ---------------------------------------------------------------------------
# Non-root user
# UID 1000 matches the default first user on most Linux desktops, which makes
# bind-mounted output directories writeable without extra chmod steps.
# ---------------------------------------------------------------------------
RUN useradd -l -m -u 1000 appcam && \
    chown -R appcam:appcam /app && \
    mkdir -p /tmp/Ultralytics && chmod 777 /tmp/Ultralytics
USER appcam

# ---------------------------------------------------------------------------
# Health check
# Interval:     how often Docker checks
# Timeout:      max time the check script may run
# Start-period: grace period after container start (model load + initial capture)
# Retries:      failures needed before marking unhealthy
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD /app/healthcheck.sh

ENTRYPOINT ["/app/entrypoint.sh"]
