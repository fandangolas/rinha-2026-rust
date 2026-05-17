#!/usr/bin/env bash
# full-sweep.sh: sweep (index, probes, instances, concurrency) inside Docker
# Usage: ./scripts/full-sweep.sh
set -euo pipefail

BENCH=/home/user/rinha-bench
URL=http://localhost:9999
N=4000
WARMUP=500
DATA=/tmp/rinha-data

wait_ready() {
  for i in $(seq 1 30); do
    curl -sf "$URL/ready" &>/dev/null && return 0
    sleep 1
  done
  echo "ERROR: stack not ready after 30s" >&2; exit 1
}

run_bench() {
  local label="$1" c="$2"
  $BENCH -url "$URL" -n "$N" -warmup "$WARMUP" -c "$c" -label "$label"
}

# Bring up a stack with specific index+probes, run bench at c=1 and c=3
bench_stack() {
  local compose="$1" index_file="$2" probes="$3" label_prefix="$4"

  docker compose -f "$compose" down --remove-orphans 2>/dev/null || true
  sleep 1

  IVF_PROBES="$probes" INDEX_PATH="/data/$index_file" \
    docker compose -f "$compose" \
      --env-file /dev/null \
      -p "sweep-$(basename $compose .yml)" \
      up -d 2>/dev/null

  wait_ready

  run_bench "$label_prefix | c=1" 1
  run_bench "$label_prefix | c=3" 3

  docker compose -f "$compose" \
    -p "sweep-$(basename $compose .yml)" \
    down 2>/dev/null || true
  sleep 1
}

cd /home/user/rinha-de-backend-2026

# Override compose to mount the index from host
# We patch by injecting a volume; simpler to use a temp override file.
make_override() {
  local index_file="$1" probes="$2"
  cat <<YAML
services:
  api1:
    volumes:
      - $DATA/$index_file:/data/index.ivf.bin:ro
    environment:
      IVF_PROBES: "$probes"
  api2:
    volumes:
      - $DATA/$index_file:/data/index.ivf.bin:ro
    environment:
      IVF_PROBES: "$probes"
YAML
}

make_override3() {
  local index_file="$1" probes="$2"
  cat <<YAML
services:
  api1:
    volumes:
      - $DATA/$index_file:/data/index.ivf.bin:ro
    environment:
      IVF_PROBES: "$probes"
  api2:
    volumes:
      - $DATA/$index_file:/data/index.ivf.bin:ro
    environment:
      IVF_PROBES: "$probes"
  api3:
    volumes:
      - $DATA/$index_file:/data/index.ivf.bin:ro
    environment:
      IVF_PROBES: "$probes"
YAML
}

OVERRIDE2=/tmp/sweep-override2.yml
OVERRIDE3=/tmp/sweep-override3.yml

run_combo() {
  local index_file="$1" probes="$2" cents="$3"

  # 2 instances
  make_override "$index_file" "$probes" > "$OVERRIDE2"
  docker compose -f docker-compose.yml -f "$OVERRIDE2" down --remove-orphans 2>/dev/null || true
  sleep 1
  docker compose -f docker-compose.yml -f "$OVERRIDE2" up -d 2>/dev/null
  wait_ready
  run_bench "2-inst ${cents}c probes=${probes} | c=1" 1
  run_bench "2-inst ${cents}c probes=${probes} | c=3" 3
  docker compose -f docker-compose.yml -f "$OVERRIDE2" down 2>/dev/null || true
  sleep 2

  # 3 instances
  make_override3 "$index_file" "$probes" > "$OVERRIDE3"
  docker compose -f docker-compose.3api.yml -f "$OVERRIDE3" down --remove-orphans 2>/dev/null || true
  sleep 1
  docker compose -f docker-compose.3api.yml -f "$OVERRIDE3" up -d 2>/dev/null
  wait_ready
  run_bench "3-inst ${cents}c probes=${probes} | c=1" 1
  run_bench "3-inst ${cents}c probes=${probes} | c=3" 3
  docker compose -f docker-compose.3api.yml -f "$OVERRIDE3" down 2>/dev/null || true
  sleep 2
}

echo "========================================"
echo "Full Docker sweep: centroids × probes × instances"
echo "========================================"

# 1000 centroids at minimum-perfect probes (2) and one higher (5)
run_combo "index.ivf.bin"  2  "1k"
run_combo "index.ivf.bin"  5  "1k"

# 5000 centroids at minimum-perfect probes (5) and one higher (10)
run_combo "index5k.ivf.bin" 5  "5k"
run_combo "index5k.ivf.bin" 10 "5k"

echo "========================================"
echo "Sweep complete."
