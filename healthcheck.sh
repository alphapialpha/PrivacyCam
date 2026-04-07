#!/usr/bin/env bash
# healthcheck.sh — called by Docker's HEALTHCHECK directive.
#
# Exits 0 (healthy) if privacy_cam.py has succeeded within the last
# HEALTH_MAX_AGE_SECONDS seconds.  Exits 1 (unhealthy) otherwise.
set -euo pipefail

HEALTH_FILE="${HEALTH_FILE:-/tmp/last_success_epoch}"
HEALTH_MAX_AGE_SECONDS="${HEALTH_MAX_AGE_SECONDS:-600}"

# No successful run yet (file missing or empty) → unhealthy.
if [[ ! -s "${HEALTH_FILE}" ]]; then
    exit 1
fi

last="$(cat "${HEALTH_FILE}" 2>/dev/null || echo 0)"
now="$(date +%s)"
age=$(( now - last ))

# Healthy if last success was within the allowed window.
(( age <= HEALTH_MAX_AGE_SECONDS ))
