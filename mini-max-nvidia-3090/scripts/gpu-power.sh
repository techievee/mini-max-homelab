#!/usr/bin/env bash
# Switch all NVIDIA GPUs between factory power/clocks and the boot-time cap from nvidia-power-limit.service.
#
#   sudo scripts/gpu-power.sh default   # each card back to its own default power limit + unlock GPU clocks
#   sudo scripts/gpu-power.sh capped    # same as the service: 250 W + clocks locked at 1350 MHz
#   scripts/gpu-power.sh status         # show current limits/clocks (no sudo needed)
#
# Only changes the running state. nvidia-power-limit.service re-applies 250 W / 1350 MHz on the next boot.
set -euo pipefail

CAP_W=${CAP_W:-250}
CAP_MHZ=${CAP_MHZ:-1350}

status() {
  nvidia-smi --query-gpu=index,pci.bus_id,name,power.limit,power.default_limit,power.max_limit,clocks.sm,clocks.max.sm \
    --format=csv
  echo
  nvidia-smi -q -d CLOCK | grep -A2 -E "^GPU |Locked|Applications Clocks Setting" | grep -vE "^--$" | head -40 || true
}

need_root() {
  [[ $EUID -eq 0 ]] || { echo "run with sudo: sudo $0 $1" >&2; exit 1; }
}

case "${1:-status}" in
  default)
    need_root default
    nvidia-smi -pm 1 >/dev/null          # persistence mode, so settings stick while no process is running
    nvidia-smi -rgc                       # unlock GPU clocks (undo --lock-gpu-clocks)
    for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      w=$(nvidia-smi -i "$i" --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
      nvidia-smi -i "$i" -pl "$w"
    done
    ;;
  capped)
    need_root capped
    nvidia-smi -pm 1 >/dev/null
    nvidia-smi -pl "$CAP_W"
    nvidia-smi --lock-gpu-clocks="$CAP_MHZ","$CAP_MHZ"
    ;;
  status) ;;
  *) echo "usage: $0 default|capped|status" >&2; exit 2 ;;
esac

echo; status
