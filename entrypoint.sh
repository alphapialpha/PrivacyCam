#!/usr/bin/env bash
# entrypoint.sh — container startup script
#
# 1. Validates configured interval.
# 2. Checks the output directory is writable.
# 3. Runs privacy_cam.py once immediately (so the first image appears right away
#    rather than waiting up to INTERVAL_MINUTES minutes).
# 4. Generates a crontab and hands off to supercronic for scheduled execution.
set -euo pipefail

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] privacy-cam: $*"; }

INTERVAL_MINUTES="${INTERVAL_MINUTES:-5}"
OUT_DIR="${OUT_DIR:-/output}"

# ── Validate INTERVAL_MINUTES ──────────────────────────────────────────────
if ! [[ "${INTERVAL_MINUTES}" =~ ^[0-9]+$ ]] || \
     (( INTERVAL_MINUTES < 1 || INTERVAL_MINUTES > 59 )); then
    log "ERROR: INTERVAL_MINUTES must be an integer from 1 to 59 (got: '${INTERVAL_MINUTES}')."
    log "       Update INTERVAL_MINUTES in your .env file."
    exit 1
fi

# ── Verify output directory is writable ───────────────────────────────────
if [[ ! -d "${OUT_DIR}" ]]; then
    if ! mkdir -p "${OUT_DIR}" 2>/dev/null; then
        log "ERROR: Output directory '${OUT_DIR}' does not exist and could not be created."
        log "       On the host, run:  mkdir -p ./output"
        log "       Then re-run:       docker compose up"
        exit 1
    fi
fi

if [[ ! -w "${OUT_DIR}" ]]; then
    log "ERROR: Output directory '${OUT_DIR}' is not writable by container UID $(id -u)."
    log "       On the host, run one of:"
    log "         chmod a+w ./output"
    log "         chown $(id -u) ./output"
    exit 1
fi

# ── Build crontab ─────────────────────────────────────────────────────────
# supercronic supports standard 5-field Vixie cron syntax.
# */N runs at every N-th minute past the hour: */5 → 0,5,10,...,55
CRON_FILE="/tmp/crontab"
printf '*/%s * * * *  /usr/local/bin/python3 /app/privacy_cam.py\n' "${INTERVAL_MINUTES}" > "${CRON_FILE}"

log "Interval : every ${INTERVAL_MINUTES} minute(s)"
log "Cron spec: */${INTERVAL_MINUTES} * * * *"

# ── Initial capture ────────────────────────────────────────────────────────
# Run once immediately so the output image is available before the first
# scheduled tick.  A failure here is non-fatal — supercronic will retry.
log "Running initial capture..."
if /usr/local/bin/python3 /app/privacy_cam.py; then
    log "Initial capture succeeded."
else
    log "WARNING: Initial capture failed. Will retry at the next cron interval."
fi

# ── Hand off to supercronic ────────────────────────────────────────────────
log "Starting scheduler (supercronic)..."
exec supercronic "${CRON_FILE}"
