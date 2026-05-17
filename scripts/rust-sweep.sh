#!/usr/bin/env bash
# rust-sweep.sh: sweep (centroids × probes) for the Rust implementation.
#
# For each configuration:
#   1. Start the Rust docker-compose stack with that index + probe count
#   2. Apply cpu.cfs_burst_us=1000 (kernel-enforced max = quota=1ms)
#   3. Run latency benchmark at c=1 with n=5000
#   4. Run recall tool at n=5000 directly against the index
#   5. Print a CSV row
#
# Output rows are appended to benchmark-results-rust.csv.
set -euo pipefail

BENCH=/home/user/rinha-bench
RECALL=/home/user/rinha-recall
URL=http://localhost:9999
DATA=/tmp/rinha-data
COMPOSE=docker-compose.rust.yml
CSV=benchmark-results-rust.csv
BURST_SCRIPT=scripts/set-cpu-burst.sh

N_BENCH=5000
N_WARMUP=500
N_RECALL=5000   # enough to surface 0.1% recall gaps reliably

cd /home/user/rinha-de-backend-2026

wait_ready() {
  for i in $(seq 1 40); do
    curl -sf "$URL/ready" &>/dev/null && return 0
    sleep 1
  done
  echo "ERROR: stack not ready after 40s" >&2; exit 1
}

run_one() {
  local index_file="$1" probes="$2"
  local cents_label="$3"   # "1k" or "5k"
  local index_path="/data/$index_file"
  local label="rust|burst1ms|${cents_label}|p${probes}|c=1"

  echo ""
  echo "════════════════════════════════════════════"
  echo "  $cents_label centroids  probes=$probes"
  echo "════════════════════════════════════════════"

  # Bring stack down cleanly
  IVF_PROBES="$probes" INDEX_PATH="$index_path" \
    docker compose -f "$COMPOSE" down --remove-orphans 2>/dev/null || true
  sleep 1

  # Start with this config
  IVF_PROBES="$probes" INDEX_PATH="$index_path" \
    docker compose -f "$COMPOSE" up -d 2>/dev/null

  wait_ready

  # Apply burst
  bash "$BURST_SCRIPT" 1000 2>/dev/null

  # --- Latency benchmark ---
  bench_out=$($BENCH -url "$URL" -n $N_BENCH -warmup $N_WARMUP -c 1 -label "$label" 2>&1)
  echo "$bench_out"

  p50=$(echo "$bench_out" | awk '/p50\.0/ {print $2}')
  p90=$(echo "$bench_out" | awk '/p90\.0/ {print $2}')
  p95=$(echo "$bench_out" | awk '/p95\.0/ {print $2}')
  p99=$(echo "$bench_out" | awk '/p99\.0/ {gsub(" ms","",$2); print $2}')
  rps=$(echo "$bench_out" | awk '/^RPS/ {print $2}')

  # --- Recall measurement ---
  echo ""
  echo "Recall (n=$N_RECALL) …"
  recall_out=$($RECALL \
    -index "${DATA}/${index_file}" \
    -probes "$probes" \
    -n $N_RECALL \
    2>/dev/null)
  echo "$recall_out"

  dec_acc=$(echo "$recall_out" | awk '/Decision accuracy/ {gsub("%","",$3); print $3}')
  fn_pct=$(echo  "$recall_out" | awk '/False negatives/  {gsub("%","",$3); print $3}')
  fp_pct=$(echo  "$recall_out" | awk '/False positives/  {gsub("%","",$3); print $3}')
  # Line: "Est. detection score component: +2126 / +3000"
  # $5 is the actual score; $NF is always +3000 (the ceiling) — don't use $NF.
  det_score=$(echo "$recall_out" | awk '/Est. detection/ {gsub(/[+]/,"",$5); print $5}')

  # Append CSV row
  echo "rust,1000,${cents_label},${probes},1,${N_BENCH},${p50},${p90},${p95},${p99},${rps},${dec_acc},${fn_pct},${fp_pct},${det_score}" \
    >> "$CSV"

  echo ""
  echo "  → p99=${p99}ms  acc=${dec_acc}%  det=${det_score}"
}

# ── Write CSV header if file is empty or missing ──────────────────────────
if [ ! -s "$CSV" ] || ! grep -q "^lang" "$CSV" 2>/dev/null; then
  echo "lang,burst_us,centroids,probes,concurrency,requests,p50_ms,p90_ms,p95_ms,p99_ms,rps,decision_acc_pct,fn_pct,fp_pct,det_score" \
    > "$CSV"
fi

# ── Sweep ─────────────────────────────────────────────────────────────────
# 1k centroids: small table (56KB fits in L2), ~3000 vec/cluster
for p in 1 2 3 5 8 10 15; do
  run_one "index.ivf.bin" "$p" "1k"
done

# 5k centroids: larger table (280KB, spills L2), ~600 vec/cluster
for p in 3 5 8 10 15 20; do
  run_one "index5k.ivf.bin" "$p" "5k"
done

# ── Tear down ─────────────────────────────────────────────────────────────
docker compose -f "$COMPOSE" down --remove-orphans 2>/dev/null || true

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Sweep complete. Results in $CSV"
echo "═══════════════════════════════════════════════════════════"
