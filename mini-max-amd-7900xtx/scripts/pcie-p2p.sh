#!/usr/bin/env bash
# Inspect / clear ACS redirect on the Broadcom PEX88096 (1000:c010) PLX switch ports, so GPU peer traffic is switched
# locally instead of going up to the CPU root port (which caps all peer traffic per board at one x16 link, ~25-28 GB/s).
#
#   sudo scripts/pcie-p2p.sh status   # ACS control per switch port + IOMMU domain of each GPU group
#   sudo scripts/pcie-p2p.sh clear <root-port>   # ACSCtl = 0 on the PEX ports under that CPU root port only
#        AMD board:    sudo scripts/pcie-p2p.sh clear 0000:40:03.1   (GPUs are in an identity IOMMU domain -> safe)
#        NVIDIA board: sudo scripts/pcie-p2p.sh clear 0000:20:03.1   only AFTER booting with iommu=pt; with a translated
#                      (DMA-FQ) domain, IOVAs could be mis-routed to a peer BAR once ACS redirect is off. The script refuses.
#   Runtime only: ACS is re-enabled on every reboot.
#
# Also needed for the NVIDIA board: kernel arg `iommu=pt` (its GPUs are in a translated IOMMU domain, so peer writes
# must be translated by the CPU IOMMU). One-time:  sudo grubby --update-kernel=ALL --args="iommu=pt"  && reboot
# Afterwards re-run: scripts/p2p-full.py (concurrent patterns should exceed one x16 link) and scripts/rccl-bench.py.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }

ports=$(lspci -d 1000:c010 -D | awk '{print $1}')
root_of() { readlink -f "/sys/bus/pci/devices/$1" | cut -d/ -f5; }
acs_bits() {  # decode ACS Control register
  local v=$((16#$1)) n=(SrcValid TransBlk ReqRedir CmpltRedir UpstreamFwd EgressCtrl DirectTrans) out=""
  for i in "${!n[@]}"; do (( v >> i & 1 )) && out+="${n[$i]}+ "; done
  echo "${out:-none}"
}

case "${1:-status}" in
  status)
    for p in $ports; do
      v=$(setpci -s "$p" ECAP_ACS+0x6.w 2>/dev/null || echo "")
      if [[ -z $v ]]; then echo "$p  (root $(root_of $p))  no ACS capability"; else echo "$p  (root $(root_of $p))  ACSCtl=0x$v  $(acs_bits "$v")"; fi
    done
    echo
    for g in $(lspci -Dn | awk '$2 ~ /^03/ {print $1}'); do   # every display controller (GPUs + BMC VGA)
      grp=$(basename "$(readlink -f /sys/bus/pci/devices/$g/iommu_group)")
      echo "$g iommu_group=$grp type=$(cat /sys/kernel/iommu_groups/$grp/type)"
    done | sort -u
    ;;
  clear)
    root=${2:?usage: $0 clear <root-port, e.g. 0000:40:03.1>}
    # Safety: every GPU under this root port must be in an identity (untranslated) IOMMU domain.
    for g in $(lspci -Dn | awk '$2 ~ /^03/ {print $1}'); do
      [[ $(root_of "$g") == "$root" ]] || continue
      t=$(cat "/sys/kernel/iommu_groups/$(basename "$(readlink -f /sys/bus/pci/devices/$g/iommu_group)")/type")
      [[ $t == identity ]] || { echo "refusing: $g under $root is in IOMMU domain '$t' (boot with iommu=pt first)" >&2; exit 1; }
    done
    n=0
    for p in $ports; do
      [[ $(root_of "$p") == "$root" ]] || continue
      if setpci -s "$p" ECAP_ACS+0x6.w >/dev/null 2>&1; then setpci -s "$p" ECAP_ACS+0x6.w=0000 && echo "$p ACSCtl cleared" && n=$((n+1)); fi
    done
    echo "cleared $n ports under $root"
    ;;
  *) echo "usage: $0 status | clear <root-port>" >&2; exit 2 ;;
esac
