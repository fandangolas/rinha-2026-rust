#!/usr/bin/env bash
# run-k6.sh — run the official k6 load test against the local Rust API stack.
#
# Prerequisites:
#   docker compose  (v2+)
#   k6              (https://k6.io/docs/get-started/installation/)
#
# Usage:
#   ./scripts/run-k6.sh              # pull pre-built image + run test
#   ./scripts/run-k6.sh --build      # build image locally (slow; needs internet)
#   IVF_PROBES=5 ./scripts/run-k6.sh # override number of IVF probe clusters
#
# NOTE: on macOS, Docker Desktop enforces cpu_period/cpu_quota through the
# Linux VM, so throttling behaviour approximates but does not perfectly match
# a bare-metal Linux runner (the official judge environment).  Use the results
# as directional signal, not ground truth.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR="${REPO_ROOT}/test"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.rust.yml"   # UDS + timing logs
TEST_DATA_URL="https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/test/test-data.json"
BUILD=0
BURST_US="${BURST_US:-20000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)  BUILD=1; shift ;;
    --tcp)    COMPOSE_FILE="${REPO_ROOT}/docker-compose.local-k6.yml"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── check prerequisites ───────────────────────────────────────────────────────
if ! command -v k6 &>/dev/null; then
  echo "k6 not found. Install from https://k6.io/docs/get-started/installation/" >&2
  exit 1
fi
if ! command -v docker &>/dev/null; then
  echo "docker not found." >&2
  exit 1
fi

mkdir -p "${TEST_DIR}"

# ── download test data (cached; ~60–80 MB) ────────────────────────────────────
if [[ ! -f "${TEST_DIR}/test-data.json" ]]; then
  echo "downloading test-data.json (~60–80 MB)…"
  curl -fsSL --progress-bar "${TEST_DATA_URL}" -o "${TEST_DIR}/test-data.json"
  echo "test-data.json ready."
else
  echo "test-data.json already present — skipping download."
fi

# ── teardown on exit ──────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "stopping services…"
  docker compose -f "${COMPOSE_FILE}" down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# ── build or pull the Rust image ──────────────────────────────────────────────
if [[ $BUILD -eq 1 ]]; then
  echo ""
  echo "building Rust image locally…"
  echo "(downloads ~3M reference vectors and builds IVF index — takes 10–20 min)"
  docker compose -f "${COMPOSE_FILE}" build
else
  echo ""
  echo "pulling nsilveira/rinha-2026-api-rust:latest…"
  docker compose -f "${COMPOSE_FILE}" pull api1 2>/dev/null || \
    docker compose -f "${COMPOSE_FILE}" pull || true
fi

# ── start the stack ───────────────────────────────────────────────────────────
echo ""
echo "starting services…"
docker compose -f "${COMPOSE_FILE}" up -d

# ── wait for ready ────────────────────────────────────────────────────────────
echo "waiting for API to be ready (up to 90s)…"
READY=0
for i in $(seq 1 90); do
  if curl -sf http://localhost:9999/ready >/dev/null 2>&1; then
    echo "  ready after ${i}s"
    READY=1
    break
  fi
  sleep 1
done

if [[ $READY -eq 0 ]]; then
  echo "API not ready after 90s; check docker compose logs:" >&2
  docker compose -f "${COMPOSE_FILE}" logs --tail=40 >&2
  exit 1
fi

# ── set CFS burst ─────────────────────────────────────────────────────────────
# On Linux this requires root; on macOS Docker Desktop the cgroup FS is inside
# the VM and the script skips gracefully.
echo ""
echo "applying cpu.cfs_burst_us=${BURST_US} µs…"
if [[ "$(uname)" == "Linux" ]]; then
  "${REPO_ROOT}/scripts/set-cpu-burst.sh" "${BURST_US}" || \
    echo "  (burst setting failed — run with sudo if needed)"
else
  "${REPO_ROOT}/scripts/set-cpu-burst.sh" "${BURST_US}" 2>&1 || true
fi

# ── run k6 ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Official k6 load test  |  ramp 1 → 900 RPS over 120s"
echo " IVF_PROBES=${IVF_PROBES:-3}  TOKIO_WORKER_THREADS=${TOKIO_WORKER_THREADS:-1}"
echo " BURST_US=${BURST_US}  METRICS=${METRICS:-false}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# k6 must run from repo root so that 'test/results.json' lands in test/.
cd "${REPO_ROOT}"
k6 run "${TEST_DIR}/test.js"

# ── print results ─────────────────────────────────────────────────────────────
echo ""
RESULTS="${TEST_DIR}/results.json"
if [[ -f "${RESULTS}" ]]; then
  echo "════ k6 results ════"
  cat "${RESULTS}"
fi

# ── per-phase timing (only meaningful when METRICS=true) ─────────────────────
if [[ "${METRICS:-false}" == "true" ]]; then
  echo ""
  echo "════ api1 /metrics ════"
  curl -sf http://localhost:9999/metrics || echo "(metrics unavailable)"
  echo "════ api2 /metrics (2nd request, round-robin) ════"
  curl -sf http://localhost:9999/metrics || echo "(metrics unavailable)"
fi

# ── HAProxy slow-request summary ─────────────────────────────────────────────
echo ""
echo "════ HAProxy Tw/Tr distribution ════"
docker compose -f "${COMPOSE_FILE}" logs haproxy 2>&1 | grep "status=" | \
awk -F' ' '
{
  for(i=1;i<=NF;i++){
    if($i~/^Tw=/) { sub("Tw=","",$i); tw=int($i) }
    if($i~/^Tr=/) { sub("Tr=","",$i); tr=int($i) }
    if($i~/^Tt=/) { sub("Tt=","",$i); tt=int($i) }
  }
  tw_b=(tw==0)?"Tw=0ms":(tw==1)?"Tw=1ms":(tw<5)?"Tw=2-4ms":"Tw≥5ms"
  tr_b=(tr==0)?"Tr=0ms":(tr==1)?"Tr=1ms":(tr<5)?"Tr=2-4ms":"Tr≥5ms"
  tw_dist[tw_b]++; tr_dist[tr_b]++
  if(tt>tt_max) tt_max=tt; n++
}
END {
  print "  Tw (HAProxy queue wait):"; for(k in tw_dist) printf "    %-12s %d\n",k,tw_dist[k]
  print "  Tr (API response time):";  for(k in tr_dist) printf "    %-12s %d\n",k,tr_dist[k]
  printf "  total=%d  max_Tt=%dms\n", n, tt_max
}' 2>/dev/null || echo "  (no HAProxy timing logs found)"
