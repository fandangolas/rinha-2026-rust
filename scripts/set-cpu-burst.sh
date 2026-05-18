#!/usr/bin/env bash
# set-cpu-burst.sh — apply cpu.cfs_burst_us to all running rinha containers.
#
# CFS burst lets a container bank unused quota during idle periods and spend it
# as a burst on the next batch of requests, so throttle stalls happen BETWEEN
# requests rather than mid-request.
#
# Optimal value: 20 000 µs (20 ms).
#   - Previous default (1 000 µs) was only 1 extra CFS period — too small to
#     absorb any realistic request burst.
#   - 20 ms covers a burst of ~50–150 simultaneous requests without stalling.
#   - Fills back up in <100 ms at normal load (~12–45% utilisation).
#
# Usage:
#   ./scripts/set-cpu-burst.sh           # default: 20 000 µs
#   ./scripts/set-cpu-burst.sh 40000     # aggressive: 40 ms
#   ./scripts/set-cpu-burst.sh 0         # disable / reset to stock behaviour
#
# Requires: root or CAP_SYS_ADMIN on Linux.
# On macOS Docker Desktop the /proc cgroup path is inside the Linux VM and is
# not visible from the host — this script prints a warning and skips silently.
set -euo pipefail

BURST_US="${1:-20000}"

set_burst() {
  local name="$1"
  local cid
  cid=$(docker ps -q --filter "name=${name}" 2>/dev/null) || return 0
  [[ -z "$cid" ]] && { echo "  [skip] ${name} — container not running"; return 0; }

  local pid
  pid=$(docker inspect "$cid" --format '{{.State.Pid}}' 2>/dev/null) || return 0
  [[ -z "$pid" || "$pid" == "0" ]] && { echo "  [skip] ${name} — no PID (stopped?)"; return 0; }

  local cgroup_file="/proc/${pid}/cgroup"
  if [[ ! -f "$cgroup_file" ]]; then
    echo "  [skip] ${name} — /proc/${pid}/cgroup not accessible"
    echo "         (Docker Desktop on macOS: cgroup FS lives inside the Linux VM)"
    return 0
  fi

  local burst_file=""

  # ── cgroup v2 unified hierarchy ───────────────────────────────────────────
  # /proc/<pid>/cgroup on v2 has a single entry: "0::/<slice>"
  if grep -qE '^0::' "$cgroup_file" 2>/dev/null; then
    local slice
    slice=$(grep '^0::' "$cgroup_file" | cut -d: -f3)
    # Candidates in order of likelihood on Docker + systemd hosts.
    for f in \
      "/sys/fs/cgroup${slice}/cpu.max.burst" \
      "/sys/fs/cgroup/docker/${cid}/cpu.max.burst" \
      "/sys/fs/cgroup/system.slice/docker-${cid}.scope/cpu.max.burst"
    do
      [[ -f "$f" ]] && { burst_file="$f"; break; }
    done
  fi

  # ── cgroup v1 fallback ───────────────────────────────────────────────────
  if [[ -z "$burst_file" ]]; then
    local cg
    cg=$(grep -E '^[0-9]+:cpu[,:]' "$cgroup_file" 2>/dev/null | head -1 | cut -d: -f3 || true)
    [[ -n "$cg" ]] && burst_file="/sys/fs/cgroup/cpu${cg}/cpu.cfs_burst_us"
  fi

  if [[ -z "$burst_file" || ! -f "$burst_file" ]]; then
    echo "  [skip] ${name} — burst file not found (kernel <5.14 or unusual cgroup layout)"
    return 0
  fi

  local old
  old=$(cat "$burst_file" 2>/dev/null || echo "?")

  # The kernel enforces burst ≤ quota (EINVAL otherwise).
  # Read cpu.max ("quota period") and cap BURST_US to the quota value.
  local cpu_max_file="${burst_file%/cpu.max.burst}/cpu.max"
  local cpu_max_file_v1="${burst_file%/cpu.cfs_burst_us}/cpu.cfs_quota_us"
  local quota="$BURST_US"
  if [[ -f "$cpu_max_file" ]]; then
    local raw_quota
    raw_quota=$(awk '{print $1}' "$cpu_max_file" 2>/dev/null || echo "max")
    [[ "$raw_quota" =~ ^[0-9]+$ ]] && quota=$(( BURST_US < raw_quota ? BURST_US : raw_quota ))
  elif [[ -f "$cpu_max_file_v1" ]]; then
    local raw_quota
    raw_quota=$(cat "$cpu_max_file_v1" 2>/dev/null || echo "0")
    [[ "$raw_quota" =~ ^[0-9]+$ && "$raw_quota" -gt 0 ]] && quota=$(( BURST_US < raw_quota ? BURST_US : raw_quota ))
  fi

  if echo "$quota" | sudo tee "$burst_file" >/dev/null 2>&1; then
    echo "  ${name}: ${old} µs → ${quota} µs  (quota-capped from ${BURST_US})   [${burst_file}]"
  else
    echo "  [fail] ${name} — could not write ${burst_file}"
  fi
}

echo "Setting cpu.cfs_burst_us=${BURST_US} µs on all rinha containers…"
for svc in haproxy api1 api2; do
  set_burst "rinha-2026-${svc}-1"
  set_burst "rinha-2026-rust-${svc}-1"
done
echo "Done."
