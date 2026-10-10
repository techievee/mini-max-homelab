#!/usr/bin/env bash
# Write the current render node of each RX 7900 XTX (by PCI address) into .env as AMD_RENDER_0..3.
# Run after any BIOS / hardware change, then `docker compose up -d --force-recreate <service>`.
set -euo pipefail
cd "$(dirname "$0")/.."
BUSES=(0000:48:00.0 0000:4e:00.0 0000:53:00.0 0000:56:00.0)   # edit if the cards move to new PCI addresses
touch .env
for i in "${!BUSES[@]}"; do
  b=${BUSES[$i]}
  [ "$(cat /sys/bus/pci/devices/$b/vendor 2>/dev/null)" = "0x1002" ] || { echo "no AMD GPU at $b" >&2; exit 1; }
  r=$(basename "$(readlink -f /dev/dri/by-path/pci-$b-render)")
  sed -i "/^AMD_RENDER_$i=/d" .env && echo "AMD_RENDER_$i=$r" >> .env
  echo "AMD_RENDER_$i=$r   ($b)"
done
