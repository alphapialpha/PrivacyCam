#!/usr/bin/env python3
"""
PrivacyCam — one-shot privacy filter for a USB webcam.

Pipeline:
  1. Capture a single frame from the configured webcam.
  2. Detect persons with YOLO and blur them.
  3. Save the blurred image to OUT_DIR/LATEST_NAME (and optionally a timestamped copy).
  4. Upload via SFTP if UPLOAD_ENABLED=true.
  5. Write a success timestamp so the healthcheck can verify liveness.

Invoked by cron (via supercronic) every INTERVAL_MINUTES minutes.
All configuration is read from environment variables — see .env.example.
"""

import base64
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Suppress Ultralytics telemetry, auto-update nags, and verbose banners before
# any ultralytics import triggers side-effects.
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("YOLO_TELEMETRY", "False")

import cv2  # noqa: E402  (must come after env setup)
import paramiko  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("privacy_cam")


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

OUT_DIR           = Path(os.environ.get("OUT_DIR", "/output"))
LATEST_NAME       = os.environ.get("LATEST_NAME", "latest.jpg")
WRITE_TIMESTAMPED = _env_bool("WRITE_TIMESTAMPED", False)

HEALTH_FILE       = Path(os.environ.get("HEALTH_FILE", "/tmp/last_success_epoch"))

UPLOAD_ENABLED    = _env_bool("UPLOAD_ENABLED", False)
UPLOAD_HOST       = os.environ.get("UPLOAD_HOST", "")
UPLOAD_PORT       = _env_int("UPLOAD_PORT", 22)
UPLOAD_USER       = os.environ.get("UPLOAD_USER", "")
UPLOAD_PASS       = os.environ.get("UPLOAD_PASS", "")
UPLOAD_HOST_KEY   = os.environ.get("UPLOAD_HOST_KEY", "")
UPLOAD_DEST_DIR   = os.environ.get("UPLOAD_DEST_DIR", "")
UPLOAD_REMOTE_NAME = os.environ.get("UPLOAD_REMOTE_NAME", "latest.jpg")


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def validate_config() -> None:
    if UPLOAD_ENABLED:
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
# 2. YOLO blur
# ---------------------------------------------------------------------------
def blur_persons(frame):
    """
    Run YOLO ObjectBlurrer on *frame* (numpy BGR array) and return the
    blurred result as a numpy BGR array.
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
# 3. Save
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
# 4. SFTP upload
# ---------------------------------------------------------------------------
def _parse_host_key(key_spec: str) -> tuple:
    """
    Parse a host-key string of the form "algorithm base64key" into
    (key_type_str, paramiko_key_object).

    Obtain the value with:
        ssh-keyscan -t ed25519 HOST | awk '{print $2, $3}'
    For non-standard ports:
        ssh-keyscan -t ed25519 -p PORT HOST | awk '{print $2, $3}'
    """
    parts = key_spec.strip().split(None, 1)
    if len(parts) != 2:
        raise ValueError(
            "UPLOAD_HOST_KEY must be 'algorithm base64key' "
            "(e.g. 'ssh-ed25519 AAAAC3...'). "
            "Obtain it with: ssh-keyscan -t ed25519 HOST | awk '{print $2, $3}'"
        )
    key_type, key_b64 = parts
    key_bytes = base64.b64decode(key_b64)

    if key_type == "ssh-rsa":
        pkey = paramiko.RSAKey(data=key_bytes)
    elif key_type == "ssh-ed25519":
        pkey = paramiko.Ed25519Key(data=key_bytes)
    elif key_type.startswith("ecdsa-sha2-"):
        pkey = paramiko.ECDSAKey(data=key_bytes)
    else:
        raise ValueError(f"Unsupported host key algorithm: {key_type!r}")

    return key_type, pkey


def upload_sftp(local_path: Path) -> None:
    """Upload *local_path* to the configured SFTP server."""
    ssh = paramiko.SSHClient()

    if UPLOAD_HOST_KEY:
        key_type, pkey = _parse_host_key(UPLOAD_HOST_KEY)
        # For non-22 ports, paramiko's known-hosts lookup uses "[host]:port".
        lookup_host = (
            UPLOAD_HOST if UPLOAD_PORT == 22 else f"[{UPLOAD_HOST}]:{UPLOAD_PORT}"
        )
        ssh.get_host_keys().add(lookup_host, key_type, pkey)
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        log.info("SFTP: host key verification enabled (%s)", key_type)
    else:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        log.warning(
            "SFTP: UPLOAD_HOST_KEY is not set — host identity is NOT verified. "
            "Acceptable for uploads to a trusted LAN machine; "
            "set UPLOAD_HOST_KEY for internet-facing servers."
        )

    remote_path = f"{UPLOAD_DEST_DIR.rstrip('/')}/{UPLOAD_REMOTE_NAME}"

    try:
        ssh.connect(
            hostname=UPLOAD_HOST,
            port=UPLOAD_PORT,
            username=UPLOAD_USER,
            password=UPLOAD_PASS,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
            # Disable SSH-key/agent auth; we use password only.
            look_for_keys=False,
            allow_agent=False,
        )
        sftp = ssh.open_sftp()
        sftp.put(str(local_path), remote_path)
        sftp.close()
        log.info("Uploaded to %s:%s", UPLOAD_HOST, remote_path)
    finally:
        ssh.close()


# ---------------------------------------------------------------------------
# 5. Health file
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
        "privacy_cam start — device=%d res=%dx%d model=%s",
        WEBCAM_DEVICE, CAPTURE_WIDTH, CAPTURE_HEIGHT, MODEL_PATH,
    )
    validate_config()

    frame   = capture_frame()
    blurred = blur_persons(frame)
    latest  = save_output(blurred)

    if UPLOAD_ENABLED:
        upload_sftp(latest)

    mark_success()
    log.info("privacy_cam done")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("privacy_cam failed: %s", exc, exc_info=True)
        sys.exit(1)
