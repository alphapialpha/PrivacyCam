# LensVeil

A self-contained Docker service that captures a still image from a USB webcam at a configurable interval, automatically detects and blurs all persons in the frame using [YOLO](https://docs.ultralytics.com/), and optionally uploads the result to a remote server via FTP or FTPS.

Designed for outdoor or publicly visible webcams where privacy regulations (e.g. GDPR) require anonymising people before publishing images.

## How it works

```
every N minutes
      │
      ▼
 capture frame          cv2.VideoCapture  →  raw JPEG in memory
      │
      ▼
 detect & blur          YOLO detection + cv2.GaussianBlur  →  blurred numpy array
      │
      ▼
 save to disk           /output/latest.jpg  (+ optional timestamped copy)
      │
      ▼
 upload (optional)      FTP / FTPS via ftplib  →  remote server
      │
      ▼
 write health file      /tmp/last_success_epoch  →  Docker healthcheck
```

The container runs a single Python script (`lensveil.py`) triggered by [supercronic](https://github.com/aptible/supercronic) — a proper cron daemon built for containers. The YOLO model (`yolov8n.pt`) is baked into the image at build time so the container works fully offline after the initial build.

## Requirements

- Linux host with a USB webcam (`/dev/video0` or similar)
- [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) (Compose v2, i.e. `docker compose`)
- Internet access during `docker compose build` (to pull the base image, pip packages, and YOLO model)

> **macOS / Windows:** The built-in webcam cannot be passed into a Docker container. Run `lensveil.py` natively with Python instead (see [Running natively](#running-natively-without-docker)).

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/AlphaPiAlpha/LensVeil.git
cd LensVeil

# 2. Create your .env from the example and edit it
cp .env.example .env
nano .env          # or your preferred editor

# 3. Find your webcam device index
ls /dev/video*     # typically /dev/video0

# 4. Create the output directory on the host
mkdir -p output

# 5. Build and start
docker compose up --build

# To run in the background:
docker compose up --build -d
docker compose logs -f
```

The first blurred image appears in `./output/latest.jpg` within a few seconds of startup. After that it refreshes every `INTERVAL_MINUTES` minutes.

## Webcam device

Check which `/dev/video*` node belongs to your webcam:

```bash
ls /dev/video*
# or for more detail:
v4l2-ctl --list-devices
```

If your webcam is at `/dev/video1`, set `WEBCAM_DEVICE=1` in `.env` **and** change the `devices:` entry in `docker-compose.yml`:

```yaml
devices:
  - /dev/video1:/dev/video1
```

To list supported resolutions for a device:

```bash
v4l2-ctl --list-formats-ext --device /dev/video0
```

## Configuration reference

All settings live in your `.env` file. Copy `.env.example` to `.env` — it is gitignored and will never be committed.

### Webcam

| Variable | Default | Description |
|---|---|---|
| `WEBCAM_DEVICE` | `0` | OpenCV device index. `0` = `/dev/video0`, `1` = `/dev/video1`, etc. |
| `CAPTURE_WIDTH` | `1280` | Requested capture width in pixels. |
| `CAPTURE_HEIGHT` | `720` | Requested capture height in pixels. |
| `WARMUP_FRAMES` | `3` | Frames discarded before the keeper is grabbed. Lets the camera settle auto-exposure and white-balance. |

### Schedule

| Variable | Default | Description |
|---|---|---|
| `INTERVAL_MINUTES` | `5` | How often to run the capture pipeline, in minutes (1–59). |

### YOLO blur

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/app/models/yolov8n.pt` | Path to the YOLO model **inside the container**. The Dockerfile pre-downloads `yolov8n.pt` to this path at build time. If you want a larger/more accurate model (e.g. `yolov8s.pt`), update this value **and** the `YOLO(...)` line in the Dockerfile, then rebuild. |
| `CLASSES` | `0` | Comma-separated COCO class IDs to detect and blur. `0` = person. [Full class list](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml). |
| `CONF` | `0.35` | Detection confidence threshold (0.0–1.0). Lower = more sensitive, more false positives. |
| `BLUR_RATIO` | `0.5` | Blur intensity applied to detected objects (0.1 = subtle, 1.0 = maximum). |

### Output

| Variable | Default | Description |
|---|---|---|
| `OUT_DIR` | `/output` | Directory inside the container where images are written. Bind-mounted from `./output` on the host. |
| `LATEST_NAME` | `latest.jpg` | Filename of the always-overwritten current image. |
| `WRITE_TIMESTAMPED` | `false` | Set to `true` to also save a timestamped copy (`blurred_20260407T120000Z.jpg`) alongside `latest.jpg`. Files accumulate — manage rotation yourself. |

### Healthcheck

| Variable | Default | Description |
|---|---|---|
| `HEALTH_FILE` | `/tmp/last_success_epoch` | Path where a Unix timestamp is written after each successful run. Read by `healthcheck.sh`. |
| `HEALTH_MAX_AGE_SECONDS` | `600` | Container is marked unhealthy if no successful run occurred within this many seconds. Recommended: at least `INTERVAL_MINUTES × 60 × 2`. Default (600 s) matches a 5-minute interval with two cycles of grace. |

### FTP / FTPS upload

| Variable | Default | Description |
|---|---|---|
| `UPLOAD_ENABLED` | `false` | Set to `true` to enable upload after each successful capture. |
| `UPLOAD_PROTOCOL` | `ftps` | Transfer protocol. `ftps` = FTP over explicit TLS (recommended). `ftp` = plain FTP — no encryption, use only on a trusted private network. |
| `UPLOAD_HOST` | — | Hostname or IP of the FTP server. |
| `UPLOAD_PORT` | `21` | FTP port. |
| `UPLOAD_USER` | — | FTP username. |
| `UPLOAD_PASS` | — | FTP password. |
| `UPLOAD_DEST_DIR` | — | Remote directory path. **Must already exist** on the server. |
| `UPLOAD_REMOTE_NAME` | `latest.jpg` | Filename written on the remote server. |
| `UPLOAD_TLS_VERIFY` | `true` | Verify the server's TLS certificate (FTPS only). Set to `false` for servers with self-signed or expired certificates — a warning is logged. |

> **What is FTPS?** It is standard FTP with TLS encryption added. In FTP client software it is usually labelled *"Explicit TLS"* or *"TLS/SSL Explicit"*. It uses port 21, just like plain FTP.

## Output files

| Path (host) | Description |
|---|---|
| `./output/latest.jpg` | Always-current blurred image, overwritten every interval. Serve this file from your web server. |
| `./output/blurred_*.jpg` | Timestamped archive copies (only when `WRITE_TIMESTAMPED=true`). |

## Checking container health

```bash
docker inspect --format='{{.State.Health.Status}}' LensVeil
# starting | healthy | unhealthy

docker compose logs -f
```

## Stopping and removing

```bash
docker compose down
```

The `./output` directory and its images remain on the host.

## Running natively (without Docker)

Useful on macOS, Windows, or for quick testing:

```bash
# Install dependencies (use opencv-python, not headless, for local use)
pip install ultralytics opencv-python

# Point OUT_DIR somewhere local
export OUT_DIR=./output
export WEBCAM_DEVICE=0
export UPLOAD_ENABLED=false

mkdir -p output
python lensveil.py
```

The YOLO model is downloaded automatically on first run and cached by Ultralytics.

## Project structure

```
LensVeil/
├── .env.example       # All config variables with documentation — copy to .env
├── .gitignore         # Ensures .env and output/ are never committed
├── .dockerignore      # Ensures .env is never baked into the Docker image
├── docker-compose.yml # Service definition — loads config from .env
├── Dockerfile         # Builds the image; pre-downloads the YOLO model
├── entrypoint.sh      # Validates config, does initial run, starts supercronic
├── healthcheck.sh     # Checked by Docker every 30 s
└── lensveil.py         # The entire pipeline: capture → blur → save → upload
```

## Changing the YOLO model

`yolov8n.pt` (nano) is the fastest and smallest model. For better detection accuracy at the cost of speed and image size:

1. Edit the `Dockerfile` — change `yolov8n.pt` in the `YOLO(...)` download line to e.g. `yolov8s.pt`.
2. Edit `.env` — set `MODEL_PATH=/app/models/yolov8s.pt`.
3. Rebuild: `docker compose build`.

Available models: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` (ascending accuracy/size).

## License

MIT — see [LICENSE](LICENSE).
