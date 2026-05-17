#!/usr/bin/env bash
# set-cpu-burst.sh: apply cpu.cfs_burst_us to all running rinha containers.
#
# CFS burst lets a container accumulate unused quota during idle periods and
# spend it as a one-shot burst on the next batch of requests. This defers
# throttling to the gap *between* requests rather than firing mid-request.
#
# Usage:  scripts/set-cpu-burst.sh [burst_us]   (default: 1000)
#         burst_us=0 to disable / reset to stock behaviour
#
# Requires: root or CAP_SYS_ADMIN (write access to /sys/fs/cgroup/cpu).
set -euo pipefail

BURST_US="${1:-1000}"

set_burst() {
  local name="$1"
  local cid
  cid=$(docker ps -q --filter "name=$name" 2>/dev/null) || return 0
  [ -z "$cid" ] && return 0

  local pid cgroup burst_file
  pid=$(docker inspect "$cid" --format '{{.State.Pid}}')
  cgroup=$(grep -E '^[0-9]+:cpu[,:]' /proc/"$pid"/cgroup | head -1 | cut -d: -f3)
  burst_file="/sys/fs/cgroup/cpu${cgroup}/cpu.cfs_burst_us"

  if [ ! -f "$burst_file" ]; then
    echo "  [skip] $name — cpu.cfs_burst_us not found (kernel too old?)"
    return 0
  fi

  local old
  old=$(cat "$burst_file")
  echo "$BURST_US" > "$burst_file"
  echo "  $name: ${old}µs → ${BURST_US}µs"
}

echo "Setting cpu.cfs_burst_us=${BURST_US} on all rinha containers…"
for svc in haproxy api1 api2; do
  set_burst "rinha-2026-${svc}-1"
  set_burst "rinha-2026-rust-${svc}-1"
done
echo "Done."
