#!/usr/bin/env bash
# bench-instances.sh — compare 2-instance vs 3-instance deployments.
#
# Prerequisites:
#   docker compose, go (for the bench binary)
#   image fandangolas/rinha-2026-api:latest must be built first
#
# Usage:
#   ./scripts/bench-instances.sh            # default: -n 4000 -c 6 -warmup 500
#   ./scripts/bench-instances.sh -n 8000 -c 9
#
# Competition-representative concurrency: the judge sends ~3 concurrent requests
# to the cluster. Because concurrency is applied to the HAProxy entry point (not
# per instance), test at c=3 and c=6 to simulate realistic and heavier load.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_BIN="${REPO_ROOT}/api/cmd/bench/bench"
N=4000
WARMUP=500
# Concurrency levels to sweep.  c=3 ≈ competition load; c=6 = stress test.
CONCURRENCIES=(3 6 9)

# ── parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) N="$2"; shift 2 ;;
    -warmup) WARMUP="$2"; shift 2 ;;
    -c) CONCURRENCIES=("$2"); shift 2 ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

# ── build bench binary ───────────────────────────────────────────────────────
echo "building bench binary…"
(cd "${REPO_ROOT}" && go build -o "${BENCH_BIN}" ./api/cmd/bench)

# ── helper: run one configuration ────────────────────────────────────────────
run_config() {
  local compose_file="$1"
  local label="$2"

  echo ""
  echo "════════════════════════════════════════"
  echo "  ${label}"
  echo "════════════════════════════════════════"

  # Bring up stack (detached).
  docker compose -f "${compose_file}" up -d

  # Wait for HAProxy to report healthy (up to 60 s).
  echo "waiting for stack to be ready…"
  for i in $(seq 1 60); do
    if curl -sf http://localhost:9999/ready >/dev/null 2>&1; then
      echo "ready after ${i}s"
      break
    fi
    sleep 1
  done

  # One sweep per concurrency level.
  for c in "${CONCURRENCIES[@]}"; do
    "${BENCH_BIN}" \
      -url http://localhost:9999 \
      -n "${N}" \
      -warmup "${WARMUP}" \
      -c "${c}" \
      -label "${label} | c=${c}"
  done

  # Tear down.
  docker compose -f "${compose_file}" down
  echo ""
}

# ── run both configurations ───────────────────────────────────────────────────
run_config "${REPO_ROOT}/docker-compose.yml"      "2 instances (0.45 CPU × 2, 160 MB × 2)"
run_config "${REPO_ROOT}/docker-compose.3api.yml" "3 instances (0.30 CPU × 3, 100 MB × 3)"

echo "════════════════════════════════════════"
echo "done. compare p99 and RPS across configs."
echo ""
echo "expected outcome:"
echo "  total CPU is fixed at 0.90 cores → RPS should be similar."
echo "  2-instance may win p99 at low-c (more CPU burst per instance)."
echo "  3-instance may win p99 at high-c (lower per-instance queue depth)."
echo "════════════════════════════════════════"
