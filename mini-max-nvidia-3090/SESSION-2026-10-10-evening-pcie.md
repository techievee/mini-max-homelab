# PCIe / P2P tuning session — 2026-10-10, ~20:00 → ~23:00

Host: ASUS Pro WS WRX80E-SAGE SE WIFI II, Threadripper PRO 5975WX, 377 GB RAM, Fedora (kernel 7.1.3-200.fc44).
GPUs: 8× RTX 3090 (4 direct-attach + 4 on a Broadcom PEX88096 PLX board) and 4× RX 7900 XTX (on a second PEX88096 board).
Companion documents: [`REPORT-2026-10-10.md`](REPORT-2026-10-10.md) (full day, §12 summarises this evening) and
[`../mini-max-amd-7900xtx/README.md`](../mini-max-amd-7900xtx/README.md) (AMD stack).

## Contents
1. [Outcome in one table](#1-outcome-in-one-table)
2. [Final configuration](#2-final-configuration)
3. [Timeline](#3-timeline)
4. [Hardware fixes](#4-hardware-fixes)
5. [BIOS / firmware](#5-bios--firmware)
6. [Kernel / OS fixes (ACS + IOMMU)](#6-kernel--os-fixes-acs--iommu)
7. [Measurements, stage by stage](#7-measurements-stage-by-stage)
8. [Collective-library tuning that did not help](#8-collective-library-tuning-that-did-not-help)
9. [AMD-specific limits](#9-amd-specific-limits)
10. [8-GPU NVIDIA: cross-root-complex P2P](#10-8-gpu-nvidia-cross-root-complex-p2p)
11. [Config and script changes](#11-config-and-script-changes)
12. [How to identify a specific card](#12-how-to-identify-a-specific-card)
13. [Open items](#13-open-items)
14. [How to re-run every test](#14-how-to-re-run-every-test)
15. [Video benchmarks with the new settings](#15-video-benchmarks-with-the-new-settings-2300000)

---

## 1. Outcome in one table

4 GPUs on one PLX switch, `scripts/p2p-full.py` (512 MiB copies) and `scripts/rccl-bench.py` (256 MiB), same cards throughout.

| | Start of evening | After reseat + riser fixes | **After kernel args (final)** |
|---|---|---|---|
| **NVIDIA PLX** link widths | 33:00.0 at **x8** | all x16 | all x16 |
| NVIDIA one pair, both directions | 25.5 GB/s | 25.5 | **52.1** |
| NVIDIA 2 pairs bidirectional / ring | 25.6 | 25.6 | **104.1** |
| NVIDIA all-to-all | 25.5 | 25.5 | **75.9** |
| NVIDIA NCCL collectives (bus bw) | 6.3–6.4 | 6.3–6.4 | **23.0–24.7** |
| **AMD PLX** uplink / card links | uplink **x8**, 48:00.0 **x8** | all x16 | all x16 |
| AMD host↔GPU per card | 13.8 GB/s | 27.6–28.1 | 27.6–28.1 |
| AMD 2 pairs bidirectional | 13.8 | 27.8 | **55.0–75.8** |
| AMD ring | 13.8 | 27.8 | **46.7–49.4** |
| AMD all-to-all | 13.8 | 26.3–26.6 | **36.8** |
| AMD RCCL collectives (bus bw) | 3.5 | 6.9–7.0 | **23.3–25.6** |

Net: collectives inside each PLX board went from 3.5–6.4 GB/s to **~23–25 GB/s (≈4–7×)**.

---

## 2. Final configuration

| Item | Value |
|---|---|
| BIOS | **1501 (2024-05-06), unchanged** — 1901 is available (see §5) |
| Kernel cmdline (added tonight) | `iommu=pt pci=disable_acs_redir=pci:1000:c010` (on top of `amd_iommu=on pcie_aspm=off`) |
| IOMMU default domain | Passthrough → all 12 GPUs in `identity` domains |
| ACS on PEX88096 ports | Redirect disabled by the kernel at boot |
| AMD PLX board | CPU root port **40:03.1**, Gen4 x16 uplink. GPUs 48/4e/53/56:00.0, render nodes renderD130–133 |
| NVIDIA PLX board | CPU root port 20:03.1, Gen4 x16. GPUs 30/33/34/37:00.0 |
| NVIDIA direct-attach | 01:00.0 (TUF), **02:00.0 (MSI SUPRIM X, was 42:00.0)**, 41:00.0 (TUF), 61:00.0 (Gaming X Trio) |
| NVLink pairs | 01↔41 and 02↔61 |
| All GPU links under load | Gen4 x16 (every card on both boards) |
| 3090 power | 250 W + clocks locked at 1350 MHz (re-applied each boot by `nvidia-power-limit.service`) |
| NCCL P2P level | `PXB` (`PHB`/`SYS` still hang, §10) |

---

## 3. Timeline

| Time | Event |
|---|---|
| ~20:00 | Investigation of "all cards show PCIe 4.0 x8". Found the **AMD PLX uplink (00:03.1↔02:00.0) trained x8**; AMD 08:00.0 and NVIDIA 33:00.0 trained x8 to their switches; other cards x16. |
| ~20:15 | Both PLX boards identified as Broadcom PEX88096 (`1000:c010`, sub `1000:a096`), same tree; the x8 ports sit at different positions on the two boards and all ports report x16 capability → physical cause, not board design. |
| ~20:30 | Slow NVIDIA card identified as **MSI SUPRIM X** (subsystem `1462:3882`, 420/450 W) on the PLX board. Found the 3090 boot service also **locks clocks at 1350 MHz**. Created `scripts/gpu-power.sh`. |
| ~20:45 | Default clocks test: 4×3090 Q4 15 s **8:44 → 6:42** (sampling 1.33× faster); per-card draw 321–366 W vs ~180 W when locked. |
| ~21:20 | **Reboot 1** (user reseated the AMD PLX board into another slot and moved the SUPRIM X from 42:00.0 into 00:03.1). AMD uplink now **x16**; AMD bandwidth doubled; bus numbers changed (AMD → 48/4e/53/56, SUPRIM X → 02). Updated 8-GPU rank order and `GPU_SELECT`. |
| ~21:30 | Found peer traffic on both boards capped at one x16 (~25.6 / ~27.8 GB/s) — IOMMU translated on NVIDIA board, ACS suspected on AMD board. |
| ~21:33 | Traced the x8 AMD card (48:00.0) by loading only that card so its fans spun up. |
| ~21:50 | **Reboot 2** (user fixed the AMD riser for 48:00.0 and the NVIDIA SUPRIM X 33:00.0). **All cards Gen4 x16**. |
| ~22:00 | Full P2P + RCCL/NCCL benchmarks, NCCL/RCCL environment sweeps (13 AMD, 9 NVIDIA settings): collectives stuck at ~6.4–7.0 GB/s regardless of settings. |
| ~22:05 | `sudo pcie-p2p.sh status` confirmed **ACSCtl=0x001d (SrcValid ReqRedir CmpltRedir UpstreamFwd)** on every PEX downstream port. Runtime clear on the AMD board did not change bandwidth. |
| ~22:15 | BIOS update instructions (1501 → 1901); kernel-parameter route chosen. |
| 22:23 | **Reboot 3** with `iommu=pt pci=disable_acs_redir=pci:1000:c010`. Peer traffic now switched locally: NVIDIA ring 104 GB/s, collectives ~23–25 GB/s on both vendors. 8-GPU `PHB`/`SYS` still hang. |
| ~22:40 | Interrupted video benchmarks cancelled; clean P2P-only test of each 4-GPU set. AMD all-to-all investigated (`HSA_ENABLE_SDMA=0` test). |

---

## 4. Hardware fixes

| # | Change (done by the user) | Before | After |
|---|---|---|---|
| H1 | AMD PLX switch board moved from the slot on root port **00:03.1** to the slot on **40:03.1** | uplink Gen4 **x8** | Gen4 **x16** |
| H2 | MSI SUPRIM X moved from the 40:03.1 slot (was 42:00.0) to the 00:03.1 slot (now 02:00.0) | x16 | x16 (so the 00:03.1 slot itself is fine; the x8 was the switch board's fit there) |
| H3 | Riser/seating fixed for AMD card **48:00.0** (switch port 45:10.0) | Gen4 **x8** (14 GB/s) | Gen4 **x16** (27.7 GB/s) |
| H4 | Seating fixed for NVIDIA **MSI SUPRIM X 33:00.0** (switch port 32:00.0) | Gen4 **x8** (13.4 GB/s) | Gen4 **x16** (25.3–26.3 GB/s) |

Diagnosis notes:
- Link width/speed read from sysfs (`current_link_width`, `max_link_width`) for the GPU **and every bridge above it** — the GPU's own
  link on RDNA3 is to its on-card switch and always reads x16; the card's real slot link is two levels up.
- Idle cards down-train (3090 → Gen2, 7900 XTX on-card port → 2.5 GT/s x1). Always measure **under load**.

---

## 5. BIOS / firmware

**No BIOS change was made tonight** (still 1501, 2024-05-06). Researched:
- Latest for Pro WS WRX80E-SAGE SE WIFI II: **1901 (2026-08-19, AGESA ChagallWS PI 1.0.0.E)**; also 1801, 1702, 1602 since 1501.
- Update: EZ Flash 3 (BIOS → Tool) or USB BIOS FlashBack (rename with BIOSRenamer to `WRX80SG2.CAP`, FAT32 stick, FlashBack port + button).
- Before updating: record settings (bifurcation, **Above 4G Decoding**, **Re-Size BAR**, SR-IOV/IOMMU, DOCP, fans, BMC); an update resets them.
- The PEX switch ACS redirect is set by **Linux**, not the BIOS (the BIOS "ACS Enable" in AMD CBS → NBIO only affects CPU root ports),
  so the kernel parameter (§6) was the actual fix. A BIOS update is optional (security/stability).

Verified unchanged and correct: resizable BAR on — every 7900 XTX exposes a **32 GiB BAR0** and every 3090 a **32 GiB BAR1**; the switch
ports' prefetchable windows match the card BARs exactly.

---

## 6. Kernel / OS fixes (ACS + IOMMU)

### The problem
- `amd_iommu=on` with the default **Translated** domain put the NVIDIA PLX GPUs in a `DMA-FQ` domain: peer writes target IOVAs the
  switch can't route, so they go up to the root complex.
- **ACS** on every PEX88096 downstream port: `ACSCtl=0x001d` = SrcValid + **ReqRedir** + **CmpltRedir** + UpstreamFwd (read with
  `sudo scripts/pcie-p2p.sh status`). Request/completion redirect forces peer traffic to the root port. Linux enables this itself when an
  IOMMU is active. Ports without an ACS capability (switch upstream ports) make the kernel merge each board into one IOMMU group.
- Effect: every pattern on a board shared one x16 link → collectives = ceiling / 4 GPUs (NVIDIA 25.5/4 = 6.4, AMD 27.8/4 = 7.0).

### Attempt 1 — runtime ACS clear (AMD board only)
`sudo scripts/pcie-p2p.sh clear 0000:40:03.1` (script refuses boards whose GPUs are not in an identity domain). **No measurable change**,
so it was replaced by the boot-time fix.

### Fix — kernel command line (persistent)
```bash
sudo grubby --update-kernel=ALL --args="iommu=pt pci=disable_acs_redir=pci:1000:c010"   # then reboot
```
- `iommu=pt` → `iommu: Default domain type: Passthrough`; all GPU groups `identity` (DMA uses physical addresses).
- `pci=disable_acs_redir=pci:1000:c010` → kernel never enables ACS redirect on Broadcom PEX880xx ports (`redir` = redirect; valid
  parameter since Linux 5.1).
- Trade-off: weaker DMA isolation between devices — only matters for VFIO/VM passthrough.
- Check after any reboot: `cat /proc/cmdline`; `sudo scripts/pcie-p2p.sh status` (GPU groups must show `type=identity`).

---

## 7. Measurements, stage by stage

### 7.1 Link state (under load)

| Card | Start | After reboot 1 | After reboot 2 (final) |
|---|---|---|---|
| AMD uplink (switch ↔ CPU) | x8 (00:03.1) | **x16** (40:03.1) | x16 |
| AMD 48:00.0 (was 08:00.0) slot link | x8 | x8 | **x16** |
| AMD 4e / 53 / 56 slot links | x16 | x16 | x16 |
| NVIDIA 33:00.0 SUPRIM X | x8 | x8 | **x16** |
| NVIDIA 30 / 34 / 37 | x16 | x16 | x16 |
| NVIDIA PLX uplink 20:03.1 | x16 | x16 | x16 |

### 7.2 AMD bandwidth progression (512 MiB copies)

| | Start | Reboot 1 (uplink x16) | Reboot 2 (all x16) | Reboot 3 (ACS off, iommu=pt) |
|---|---|---|---|---|
| Host→GPU, x16 cards | 13.8 | 27.6 | 27.7 | 27.7 |
| Host→GPU, 48:00.0 | 13.8 | 14.2 | **27.7** | 27.7 |
| One pair, one direction | 13.8 | 14.0–24.0 | 15.8–19.2 | 15.8–19.2 |
| One pair, both directions | — | 22.8–24.8 | 22.8–24.8 | **25.0–34.3** |
| 2 disjoint pairs | 13.8 | 27.6 | 27.8 | **31.5–47.9** |
| 2 pairs bidirectional | 13.8 | 27.8 | 27.8 | **55.0–75.8** |
| Ring | 13.8 | 27.8 | 27.8 | **46.7–49.4** |
| All-to-all | — | 26.6 | 26.3 | **36.8** |
| RCCL all-reduce (bus bw) | 3.5 | 7.0 | 6.9 | **25.6** |
| RCCL all-gather / reduce-scatter / all-to-all | — | — | 6.7–7.0 | **23.3–25.0** |
| Latency 4 KiB | — | — | 84–100 µs | 82–101 µs |

Ranges are across pairs and across repeated runs; AMD results vary more than NVIDIA's (e.g. "2 pairs bidirectional" 75.8 in one run, 55.0 in another).

### 7.3 NVIDIA PLX bandwidth progression (512 MiB copies)

| | Start | Reboot 2 (all x16) | Reboot 3 (final) |
|---|---|---|---|
| Host↔GPU 33:00.0 | 13.4 / 13.2 | **25.3 / 26.3** | 25.3 / 26.3 |
| One pair, one direction | 13.2 (to 33) / 25.5 | 25.5 | **26.3** |
| One pair, both directions | — | 25.5 | **52.1** |
| 2 disjoint pairs / 2 pairs bidirectional / ring | 25.5–25.6 | 25.6 | **52.6 / 104.1 / 104.1** |
| All-to-all | — | 25.5 | **75.9** |
| One-to-all / all-to-one | — | 25.5 / 25.5 | 26.3 / 26.3 |
| NCCL all-reduce / all-gather / reduce-scatter / all-to-all (bus bw) | — | 6.3–6.4 | **24.7 / 23.8 / 23.1 / 23.0** |
| Latency 4 KiB | — | 29–38 µs | 27–35 µs |

### 7.4 3090 power / clocks (from the start of the evening)

Boot service `/etc/systemd/system/nvidia-power-limit.service` runs `nvidia-smi -pl 250` **and** `--lock-gpu-clocks=1350,1350`.

| 4×3090 Q4, 1344×768 | Capped (250 W, 1350 MHz) | Default (factory limits, unlocked) |
|---|---|---|
| 5 s clip sampling | 73 s (18.2 s/step), cards drew 173–196 W | 55 s (13.8 s/step), 318–352 W, ~1930 MHz |
| **15 s clip total (N3)** | **8:44** (sampling 332 s) | **6:42** (sampling 249 s), 321–366 W per card, 1.40 kW for 4 cards |
| Output | identical video (SSIM 1.000) | |

The clock lock, not the 250 W cap, was the limit. `scripts/gpu-power.sh default|capped|status` switches at runtime (sudo).

---

## 8. Collective-library tuning that did not help

Before the kernel fix, 256 MiB, 4 GPUs (bus bandwidth, GB/s):

| Setting | AMD RCCL | NVIDIA NCCL |
|---|---|---|
| baseline (AMD 2 channels, NVIDIA 4) | 6.9–7.0 | 6.3–6.4 |
| `NCCL_P2P_LEVEL=PXB` (AMD) | 6.9–7.0 | — |
| `NCCL_MIN/MAX_NCHANNELS=4` / `=8` | 7.0 / 7.0 | — / 6.4 |
| `NCCL_MIN/MAX_P2P_NCHANNELS=8` | 6.9–7.0 | 6.4 |
| `NCCL_BUFFSIZE=16777216` | 6.9–7.0 | 6.4 |
| `NCCL_PROTO=Simple` / `LL128` | 6.9–7.0 / 6.9–7.0 | 6.3–6.4 / 6.0 |
| `NCCL_ALGO=Tree` | 6.9–7.0 | fails |
| `NCCL_P2P_USE_CUDA_MEMCPY=1` (copy engine) | 5.9–6.7 | 5.2–6.2 |
| copy engine + 8 channels (AMD) | 6.6–6.9 | — |
| `HSA_FORCE_FINE_GRAIN_PCIE=0` (AMD) | 6.9–7.0 | — |
| `NCCL_P2P_DISABLE=1` (shared memory) | 2.9–4.8 | 2.5–2.8 |

Conclusion: the libraries were already at the physical limit of the root-port route. Only the routing change (§6) helped.

---

## 9. AMD-specific limits

- **Per-card peer DMA ≈ 18–19 GB/s total.** One-to-all 18.4 GB/s, all-to-one 19.2 GB/s. When a card sends and receives at once the two
  directions share that budget (`amd-smi topology`: **BI-DIRECTIONAL = F** for every pair), so all-to-all ≈ 4 × ~9.2 = **36.8 GB/s**
  (NVIDIA: 26 out + 26 in per card, all-to-all 75.9).
- `HSA_ENABLE_SDMA=0` (shader copies instead of SDMA engines) is **worse**: all-to-all 26.9, ring 42.5, pair-bidir 17.4–26.1 →
  keep `HSA_ENABLE_SDMA=1` (already set in `../mini-max-amd-7900xtx/docker-compose.yml`).
- Latency ~3× NVIDIA (82–101 µs vs 27–35 µs per 4 KiB copy incl. launch + sync).
- `amd-smi topology`: access ENABLED, link type PCIE, hops 2, DMA T, coherency NC for all pairs.
- **ROCm 7.14 vs ROCm 10 (tested ~23:00):** same 4 cards, same scripts. The current stack image `mini-max-comfyui:rocm`
  (torch 2.12.0+rocm7.14.1) vs `rocm/vllm:rocm10.0.0…` (torch 2.12.0+rocm10.0.0, HIP 7.15, RCCL 2.30.4 — same RCCL version).
  Two repetitions each, interleaved:

  | Pattern / op | ROCm 7.14 (rep1, rep2) | ROCm 10 (run0, rep1, rep2) |
  |---|---|---|
  | one-to-all / all-to-one | 18.4 / 19.2, 18.4 / 19.2 | 18.4 / 19.2 (all three) |
  | 2 disjoint pairs | 47.9–56.0, 47.9 | 47.9 (all three) |
  | 2 pairs bidirectional | 90.5, 83.1 | 86.9, 83.1, 83.1 |
  | ring | 81.8, 53.1 | 73.5, 55.1, 48.1 |
  | all-to-all | 42.1, 36.1 | 39.9, 36.8, 36.7 |
  | RCCL 256 MiB all_reduce / all_gather / reduce_scatter / all_to_all | 25.6–25.7 / 25.0 / 23.4 / 23.7 | 23.3–25.6 / 22.9–25.0 / 21.4–23.4 / 23.6–23.7 |
  | RCCL 64 MiB (same four) | 13.8–18.2 / 13.2–17.6 / 12.8–16.2 / 21.9 | 13.6–14.6 / 12.9–14.3 / 13.2–13.7 / 21.5–21.9 |

  **No meaningful difference:** run-to-run variance (e.g. ring 48–82 GB/s on both) is larger than any version gap. The per-card limits
  (18.4 out / 19.2 in) are identical, so they come from the card/kernel driver (amdgpu), not the ROCm user-space runtime. Keep ROCm 7.14.

---

## 10. 8-GPU NVIDIA: cross-root-complex P2P

NCCL all-reduce, 8×3090 (`CUDA_VISIBLE_DEVICES=6,0,1,7,2,3,4,5` → ranks 41, 01, 02, 61, 30, 33, 34, 37):

| `NCCL_P2P_LEVEL` | Before kernel fix | After kernel fix |
|---|---|---|
| `PXB` | 5.5 GB/s | **6.9 GB/s** |
| `PHB` | hangs | **still hangs** |
| `SYS` | hangs | **still hangs** |

P2P through the Threadripper root complex is broken independently of IOMMU/ACS; the three NODE/PHB hops of the 8-GPU ring use
shared memory and cap it. **Per the user, the 8-GPU NVIDIA configuration is not used** (4 cards on PLX + 4 direct-attach can't run as one set
efficiently); benchmarking now targets the 4-GPU PLX sets of each vendor.

---

## 11. Config and script changes

| File | Change |
|---|---|
| `docker-compose.yml` (`comfyui-cuda8`) | `CUDA_VISIBLE_DEVICES` `0,5,6,7,1,2,3,4` → **`6,0,1,7,2,3,4,5`** (new PCI order after reboot 1); UUID list/comments updated (02:00.0) |
| `user/default/api_workflows/bench_n2/n3/pw_q1.json` | `GPU_SELECT` `1,2,3,4` → **`2,3,4,5`** (PLX cards' new PCI indices) |
| `../mini-max-amd-7900xtx/docker-compose.yml` | comments: AMD GPUs now 48/4e/53/56 (render nodes unchanged renderD130–133) |
| `../mini-max-amd-7900xtx/scripts/bench.py` | sysfs card comment updated; output check now also flags **noise** (KB/frame) and auto-restarts on bf16↔GGUF switches |
| `scripts/gpu-power.sh` (new) | `default` / `capped` / `status` for 3090 power + clock lock |
| `scripts/pcie-p2p.sh` (new, both projects) | ACS status per PEX port + GPU IOMMU domains; `clear <root-port>` with identity-domain safety check |
| `scripts/p2p-full.py` (new, both projects) | per-pair uni/bidirectional (16 & 512 MiB), latency, concurrent patterns; syncs every involved device |
| `scripts/rccl-bench.py` (new, both projects) | all_reduce / all_gather / reduce_scatter / all_to_all bus bandwidth, `BENCH_MIBS` env |
| `scripts/bench/pwtest.sh`, `scripts/bench/nvrun.sh` (new) | NVIDIA video benchmark runner with per-GPU power/clock sampling and output check (kept in the project; `/tmp` is wiped at reboot) |

---

## 12. How to identify a specific card

- **Make/model:** PCI subsystem ID + factory power limits, e.g. `lspci -nn -s 33:00.0 -v`, `nvidia-smi -q -i 00000000:33:00.0`.
  `1462:3882` + 420/450 W = MSI SUPRIM X; `1043:87af` + 390/480 W = ASUS ROG Strix; `1043:87b3` + 350/375 W = ASUS TUF;
  `1462:3884` + 370/380 W = MSI Gaming X Trio; `10de:1454` = NVIDIA-branded; `1eae:7901` = XFX RX 7900 XTX.
- **Air-cooled AMD card:** load only that card (container with just its `/dev/dri/renderD13x`); its fans spin up while the others stay at
  0 RPM (48:00.0 went 492 → 945 RPM within a minute).
- **Water-cooled NVIDIA card:** count outputs (ASUS Strix has 2× HDMI; reference/SUPRIM have 1×), or plug a monitor in and watch
  `/sys/class/drm/card*-*/status` — the DRM connector maps to the PCI address (`nvidia-smi display_mode` is deprecated on this driver).

---

## 13. Open items

1. Optional BIOS update 1501 → 1901 (record and restore Above 4G / Re-Size BAR / bifurcation settings).
2. Decide whether to keep `--lock-gpu-clocks=1350` in `nvidia-power-limit.service` (costs ~25 % throughput on 4×3090).
3. ~~Re-run the video benchmarks~~ done, see §15.
4. AMD peer DMA (~18–19 GB/s per card, no bidirectional) is a card/kernel-driver limit; ROCm 10 user space gives the same numbers (§9).
5. 8-GPU `PHB`/`SYS` hang (cross-root-complex P2P) — no fix found; not needed for the 4-GPU sets.

---

## 14. How to re-run every test

```bash
# Topology / links (no root): GPU + all parent bridges, under load for real values
for d in /sys/bus/pci/devices/*; do ...; done        # see REPORT §12 commands, or:
nvidia-smi --query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi topo -m

# ACS / IOMMU (root)
sudo <repo>/mini-max-nvidia-3090/scripts/pcie-p2p.sh status

# NVIDIA PLX set (4-GPU service = the 4 PLX cards)
cd <repo>/mini-max-nvidia-3090
docker compose run --rm --entrypoint python comfyui scripts/p2p-full.py
docker compose run --rm --entrypoint torchrun comfyui --nproc_per_node 4 scripts/rccl-bench.py

# AMD PLX set
cd <repo>/mini-max-amd-7900xtx
docker compose run --rm --entrypoint python comfyui scripts/p2p-full.py
docker compose run --rm --entrypoint torchrun comfyui --nproc_per_node 4 scripts/rccl-bench.py
docker compose run --rm --entrypoint bash comfyui -c 'amd-smi topology'

# 3090 power/clocks
sudo <repo>/mini-max-nvidia-3090/scripts/gpu-power.sh default   # or capped / status
```
Stop containers that hold the GPUs first (`mini-max-nvidia-3090-cuda8`, `mini-max-amd-7900xtx-bench`), and give `docker compose run` an empty stdin
(`</dev/null`) when looping over it in a script.

## 15. Video benchmarks with the new settings (~23:00–00:00)

After reboot 4 (22:58; same topology, `iommu=pt pci=disable_acs_redir` still active). 4-GPU PLX sets only, run in parallel (separate PSUs),
same prompt/seed/settings as the earlier runs, 1344×768, 4-step turbo LoRA. NVIDIA at the boot cap (250 W / 1350 MHz) like the "before"
NVIDIA numbers; AMD stock. Every output checked (black + noise) and eyeballed before the next run.

| Set | Run | Before (sampling, per step / total) | **After (sampling, per step / total)** | Sampling gain |
|---|---|---|---|---|
| **AMD 4×7900 XTX** | s1 bf16+FSDP, 5 s | 156 s, 39.0 / 291 s | **54 s, 13.5 / 262 s** | **2.9×** |
| | s2 Q4, 5 s | 90 s, 22.5 / 170 s | **64 s, 16.0 / 145 s** | **1.41×** |
| | F0 bf16+FSDP, 15 s | 440 s, 110 / 11:06 | **349 s, 87.2 / 9:24** | **1.26×** |
| | F1 Q4, 15 s | 393 s, 98.2 / 8:50 | **330 s, 82.5 / 7:50** | **1.19×** |
| **NVIDIA 4×3090** | Q4, 5 s | 73 s, 18.2 / 148 s | **63 s, 15.8 / 178 s*** | 1.16× |
| | N3 Q4, 15 s | 332 s, 83 / 8:44 | **307 s, 76.8 / 8:13** | 1.08× |
| | N2 bf16+FSDP, 15 s | 344 s, 86 / 9:31 | **308 s, 77.0 / 10:35*** | 1.12× |

\* Total includes a cold start after the container restart (bf16 text encoder 189 s vs 87 s warm); compare the sampling column.

- Gains scale with how much the config communicates: AMD bf16+FSDP on short clips gained most (2.9×), NVIDIA Q4 15 s least (1.08×).
- AMD gained more than NVIDIA because its old path was worse (x8 uplink + ACS → 13.8 GB/s; NVIDIA was already x16 → 25.6 GB/s).
- **4 vs 4, best config, 15 s:** AMD Q4 **7:50** vs NVIDIA Q4 capped **8:13**. With the 3090 clock lock removed, NVIDIA was 6:42 before
  this change, so it would likely still lead (estimate, not re-run).
- Peak VRAM: AMD 24.1–24.5 GB on GPU 0 (text encoder), ~20–21 GB on the others; NVIDIA 18.7 (Q4) / 23.8 GB (bf16). No swap.
- Logs: `output/bench_acsoff_nvidia.log`, `../mini-max-amd-7900xtx/output/bench_acsoff_amd.log`, AMD rows in `../mini-max-amd-7900xtx/output/bench/results.jsonl`.
