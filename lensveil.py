#!/usr/bin/env python3
"""
LensVeil — one-shot privacy filter for a USB webcam.

Pipeline:
  1. Capture a single frame from the configured webcam.
  2. Detect persons with YOLO and blur them.
  3. Save the blurred image to OUT_DIR/LATEST_NAME (and optionally a timestamped copy).
  4. Upload via FTP/FTPS if UPLOAD_ENABLED=true.
  5. Write a success timestamp so the healthcheck can verify liveness.

Invoked by cron (via supercronic) every INTERVAL_MINUTES minutes.
All configuration is read from environment variables — see .env.example.
"""

import ftplib
import logging
import ssl
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Suppress Ultralytics telemetry, auto-update nags, and verbose banners before
# any ultralytics import triggers side-effects.
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("YOLO_TELEMETRY", "False")

import cv2  # noqa: E402  (must come after env setup)
from ultralytics import YOLO  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("lensveil")


# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in ("true", "1", "yes")


WEBCAM_DEVICE     = _env_int("WEBCAM_DEVICE", 0)
CAPTURE_WIDTH     = _env_int("CAPTURE_WIDTH", 1280)
CAPTURE_HEIGHT    = _env_int("CAPTURE_HEIGHT", 720)
WARMUP_FRAMES     = _env_int("WARMUP_FRAMES", 3)

MODEL_PATH        = os.environ.get("MODEL_PATH", "/app/models/yolov8n.pt")
CLASSES           = [int(c.strip()) for c in os.environ.get("CLASSES", "0").split(",")]
CONF              = _env_float("CONF", 0.25)
BLUR_RATIO        = _env_float("BLUR_RATIO", 0.5)

BLUR_ENABLED      = _env_bool("BLUR_ENABLED", True)
FLIP_VERTICAL     = _env_bool("FLIP_VERTICAL", False)
FLIP_HORIZONTAL   = _env_bool("FLIP_HORIZONTAL", False)

OUT_DIR           = Path(os.environ.get("OUT_DIR", "/output"))
LATEST_NAME       = os.environ.get("LATEST_NAME", "latest.jpg")
WRITE_TIMESTAMPED = _env_bool("WRITE_TIMESTAMPED", False)

HEALTH_FILE       = Path(os.environ.get("HEALTH_FILE", "/tmp/last_success_epoch"))

UPLOAD_ENABLED    = _env_bool("UPLOAD_ENABLED", False)
UPLOAD_PROTOCOL   = os.environ.get("UPLOAD_PROTOCOL", "ftps").lower()
UPLOAD_HOST       = os.environ.get("UPLOAD_HOST", "")
UPLOAD_PORT       = _env_int("UPLOAD_PORT", 21)
UPLOAD_USER       = os.environ.get("UPLOAD_USER", "")
UPLOAD_PASS       = os.environ.get("UPLOAD_PASS", "")
UPLOAD_TLS_VERIFY  = _env_bool("UPLOAD_TLS_VERIFY", True)
UPLOAD_DEST_DIR   = os.environ.get("UPLOAD_DEST_DIR", "")
UPLOAD_REMOTE_NAME = os.environ.get("UPLOAD_REMOTE_NAME", "latest.jpg")


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def validate_config() -> None:
    if UPLOAD_ENABLED:
        if UPLOAD_PROTOCOL not in ("ftp", "ftps"):
            raise ValueError(
                f"UPLOAD_PROTOCOL must be 'ftp' or 'ftps' (got: {UPLOAD_PROTOCOL!r})"
            )
        missing = [
            name for name, val in [
                ("UPLOAD_HOST",     UPLOAD_HOST),
                ("UPLOAD_USER",     UPLOAD_USER),
                ("UPLOAD_PASS",     UPLOAD_PASS),
                ("UPLOAD_DEST_DIR", UPLOAD_DEST_DIR),
            ] if not val
        ]
        if missing:
            raise ValueError(
                f"UPLOAD_ENABLED=true but the following variables are empty: "
                f"{', '.join(missing)}"
            )

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"YOLO model not found at {MODEL_PATH!r}. "
            "The Dockerfile pre-downloads it; rebuild the image if missing."
        )


# ---------------------------------------------------------------------------
# 1. Capture
# ---------------------------------------------------------------------------
def capture_frame():
    """Open the webcam, discard warmup frames, and return a single BGR frame."""
    cap = cv2.VideoCapture(WEBCAM_DEVICE)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam at device index {WEBCAM_DEVICE}. "
            "Check that /dev/video0 is passed through in docker-compose.yml "
            "and the container has the right devices: entry."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    # Discard warmup frames so auto-exposure / white-balance can settle.
    for _ in range(WARMUP_FRAMES):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Webcam returned an empty frame after warmup.")

    h, w = frame.shape[:2]
    log.info("Captured frame: %dx%d from device %d", w, h, WEBCAM_DEVICE)
    return frame


# ---------------------------------------------------------------------------
# 2. Flip
# ---------------------------------------------------------------------------
def apply_flips(frame):
    """Apply vertical and/or horizontal flip if configured."""
    if FLIP_VERTICAL:
        frame = cv2.flip(frame, 0)
    if FLIP_HORIZONTAL:
        frame = cv2.flip(frame, 1)
    return frame


# ---------------------------------------------------------------------------
# 3. YOLO blur
# ---------------------------------------------------------------------------
def blur_persons(frame):
    """
    Run YOLO detection on *frame* (numpy BGR array), blur detected regions
    with cv2.GaussianBlur, and return the result.
    """
    model = YOLO(MODEL_PATH)
    results = model(frame, conf=CONF, classes=CLASSES, verbose=False)[0]

    out = frame.copy()
    for box in results.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = box
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        k = max(3, int(min(roi.shape[:2]) * BLUR_RATIO))
        if k % 2 == 0:
            k += 1
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    log.info("YOLO blur complete (%d region(s))", len(results.boxes))
    return out


# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
def save_output(blurred_frame) -> Path:
    """Write the blurred frame to OUT_DIR. Returns the path of latest.jpg."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = OUT_DIR / LATEST_NAME

    if not cv2.imwrite(str(latest_path), blurred_frame):
        raise RuntimeError(f"cv2.imwrite failed for {latest_path}")
    log.info("Saved: %s", latest_path)

    if WRITE_TIMESTAMPED:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ts_path = OUT_DIR / f"blurred_{ts}.jpg"
        if cv2.imwrite(str(ts_path), blurred_frame):
            log.info("Saved timestamped copy: %s", ts_path.name)
        else:
            log.warning("Failed to write timestamped copy: %s", ts_path)

    return latest_path


# ---------------------------------------------------------------------------
# 5. FTP / FTPS upload
# ---------------------------------------------------------------------------
def upload_ftp(local_path: Path) -> None:
    """Upload *local_path* via FTP or FTPS (explicit TLS) depending on UPLOAD_PROTOCOL."""
    if UPLOAD_PROTOCOL == "ftps":
        if not UPLOAD_TLS_VERIFY:
            log.warning(
                "FTP: UPLOAD_TLS_VERIFY=false — server TLS certificate is NOT verified. "
                "Acceptable for private/LAN servers with self-signed certificates; "
                "not recommended for internet-facing servers."
            )
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        else:
            ctx = ssl.create_default_context()
        ftp = ftplib.FTP_TLS(context=ctx)
    else:
        ftp = ftplib.FTP()

    try:
        ftp.connect(UPLOAD_HOST, UPLOAD_PORT, timeout=15)
        ftp.login(UPLOAD_USER, UPLOAD_PASS)
        if UPLOAD_PROTOCOL == "ftps":
            ftp.prot_p()  # upgrade data channel to TLS
        if UPLOAD_DEST_DIR:
            ftp.cwd(UPLOAD_DEST_DIR)
        with open(local_path, "rb") as f:
            ftp.storbinary(f"STOR {UPLOAD_REMOTE_NAME}", f)
        log.info(
            "Uploaded to %s://%s%s/%s",
            UPLOAD_PROTOCOL, UPLOAD_HOST, UPLOAD_DEST_DIR, UPLOAD_REMOTE_NAME,
        )
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()


# ---------------------------------------------------------------------------
# 6. Health file
# ---------------------------------------------------------------------------
def mark_success() -> None:
    """Write the current Unix epoch to HEALTH_FILE for the healthcheck."""
    try:
        HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEALTH_FILE.write_text(str(int(datetime.now(timezone.utc).timestamp())))
    except OSError as exc:
        log.warning("Could not write health file %s: %s", HEALTH_FILE, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info(
        "lensveil start — device=%d res=%dx%d model=%s",
        WEBCAM_DEVICE, CAPTURE_WIDTH, CAPTURE_HEIGHT, MODEL_PATH,
    )
    validate_config()

    frame   = capture_frame()
    frame   = apply_flips(frame)
    blurred = blur_persons(frame) if BLUR_ENABLED else frame
    latest  = save_output(blurred)

    if UPLOAD_ENABLED:
        upload_ftp(latest)

    mark_success()
    log.info("lensveil done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("lensveil failed: %s", exc, exc_info=True)
        sys.exit(1)
