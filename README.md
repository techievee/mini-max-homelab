# mini-max-homelab

MiniMax H3 (text → video + stereo audio) in ComfyUI + Raylight on a homelab box with 8× RTX 3090 and 4× RX 7900 XTX.

| Folder | Stack | Port |
|---|---|---|
| [`mini-max-amd-7900xtx/`](mini-max-amd-7900xtx) | ROCm, 4× RX 7900 XTX on a PEX88096 PCIe switch | 8190 (`comfyui`), 8192 (`comfyui-bench`) |
| [`mini-max-nvidia-3090/`](mini-max-nvidia-3090) | CUDA, 4× RTX 3090 (PLX or direct-attach) or all 8 (`cuda8`) | 8189, 8191 (`cuda8`) |

Benchmarks, PCIe/P2P findings and workarounds: [`mini-max-nvidia-3090/REPORT-2026-10-10.md`](mini-max-nvidia-3090/REPORT-2026-10-10.md).

## GPU topology and P2P

Host: ASUS Pro WS WRX80E-SAGE SE WIFI II, Threadripper PRO 5975WX (one NUMA node, four PCIe root complexes 00/20/40/60),
all GPUs PCIe Gen4 x16. Two Broadcom **PEX88096** switch boards: one for 4× RTX 3090, one for 4× RX 7900 XTX.

```
 Root complex 00 ─┬─ 00:01.1 ── 01:00.0  RTX 3090 (ASUS TUF)          NVLink A
                  └─ 00:03.1 ── 02:00.0  RTX 3090 (MSI SUPRIM X)      NVLink B
 Root complex 20 ─── 20:03.1 ── PEX88096 ─┬─ 30:00.0  RTX 3090
                                          ├─ 33:00.0  RTX 3090  ┐ same switch sub-port (PIX)
                                          ├─ 34:00.0  RTX 3090  ┘
                                          └─ 37:00.0  RTX 3090
 Root complex 40 ─┬─ 40:01.1 ── 41:00.0  RTX 3090 (ASUS TUF)          NVLink A  (bridge to 01)
                  └─ 40:03.1 ── PEX88096 ─┬─ 48:00.0  RX 7900 XTX
                                          ├─ 4e:00.0  RX 7900 XTX
                                          ├─ 53:00.0  RX 7900 XTX
                                          └─ 56:00.0  RX 7900 XTX
 Root complex 60 ─── 60:01.1 ── 61:00.0  RTX 3090 (MSI Gaming X Trio) NVLink B  (bridge to 02)
```

`nvidia-smi` index (PCI order): 0 = 01, 1 = 02, 2 = 30, 3 = 33, 4 = 34, 5 = 37, 6 = 41, 7 = 61. NVLink pairs: **0↔6** and **1↔7**.
The compose files select cards by UUID (`.env.example`), so a reseat that renumbers buses can't silently swap cards.

### How the GPUs talk to each other

| Path (`nvidia-smi topo` / NCCL name) | Pairs | Mechanism | Measured (512 MiB copies) |
|---|---|---|---|
| **NV4** (NVLink bridge) | 01↔41, 02↔61 | 4 NVLink lanes, no PCIe | 52.8 GB/s one way, ~101 GB/s both ways |
| **PIX** (same switch port) | 33↔34 | P2P DMA through one switch level | as PXB below |
| **PXB** (through the switch) | 3090s 30/33/34/37 | P2P DMA inside the PEX88096, never reaches the CPU | 26.3 one way, 52.1 both ways, ring/2 pairs 104 GB/s |
| **PXB** (through the switch) | 7900 XTX 48/4e/53/56 | same, via SDMA engines (`HSA_ENABLE_SDMA=1`) | 15.8–19.2 one way; per card ~18.4 out / 19.2 in **total** (can't send + receive at full rate) |
| **PHB** (same root complex) | 01↔02 | through the CPU host bridge | copies work (25 GB/s one way, only 26 both ways); **NCCL P2P hangs** |
| **NODE / SYS** (across root complexes) | e.g. 01↔61, PLX↔direct | through the CPU fabric | copies 25–26 one way; **NCCL P2P hangs** |
| **SHM** (no P2P) | fallback | via pinned host memory | collectives ~2.5–4.4 GB/s |

Collectives (bus bandwidth, 256 MiB, 4 GPUs): NVIDIA PLX NCCL **24.7 GB/s**; AMD RCCL **25.6 GB/s**; NVIDIA direct-attach 12.7–14.8
(NVLink pairs P2P, the rest SHM); 8× 3090 together only ~6.9 (the three cross-root hops fall back to SHM).

### Required host settings

```
# kernel command line (GRUB)
iommu=pt pci=disable_acs_redir=pci:1000:c010
```
- `pci=disable_acs_redir=pci:1000:c010`: Linux enables **ACS redirect** on the PEX88096 ports by default, which forces every
  peer transfer up to the CPU root port and back. With it disabled, peer traffic stays inside the switch: collectives went from
  ~6–7 to ~23–26 GB/s on both boards (~4×).
- `iommu=pt`: GPUs in the identity IOMMU domain, so peer DMA addresses aren't translated.
- BIOS: Above 4G Decoding + Re-Size BAR on (32 GiB BAR1 per 3090). 01 and 02 currently only expose 256 MiB (old/non-ReBAR VBIOS).
- Without the kernel args the stacks still run, just ~4× slower collectives.

### `NCCL_P2P_LEVEL` per GPU set (RCCL reads the same variable)

| Set | Use | Why |
|---|---|---|
| NVIDIA PLX 30/33/34/37 | **`PXB`** | all pairs inside the switch |
| NVIDIA direct 01/02/41/61 | **`NVL`** | P2P only over the two NVLink bridges, SHM for the rest (same as NCCL's default here) |
| AMD 48/4e/53/56 | **`PXB`** | all pairs inside the switch |
| any | never `PHB` / `SYS` | P2P through the CPU root complexes hangs on this host |
| AMD | not `PIX` | the PEX88096 shows up as more than one switch level, so `PIX` turns P2P off for every pair (measured 3.6 vs 25.6 GB/s) |

Known issue: slots 01/02 (root complex 00) log corrected PCIe errors (BadTLP / replay timeouts), and the host hard-reset once under
load on that pair. Prefer the PLX sets; if 01/02 are needed, set those slots to Gen3 in the BIOS and check the risers.

## Models: Hugging Face cache + symlinks

The weights (~131 GB) live in the standard Hugging Face cache (`~/.cache/huggingface/hub`), not in this repo.
`mini-max-amd-7900xtx/models/` only holds symlinks in the layout ComfyUI expects, and both stacks mount it
(the NVIDIA stack via `MODELS_DIR=../mini-max-amd-7900xtx/models`). Both compose files also mount the HF cache **read-only at the same
absolute path** inside the containers, so the symlinks resolve there too.

`scripts/hf-models.py` (run on the host; needs `pip install -U huggingface_hub`, and `HF_TOKEN` only for gated repos):

```bash
python3 scripts/hf-models.py status                 # every model: linked / cached / plain file / missing
python3 scripts/hf-models.py download               # fetch the "default" set into the HF cache, then link it
python3 scripts/hf-models.py download q4            # only the Q4 GGUF pair
python3 scripts/hf-models.py download dit-fp8 te-nvfp4   # single files by key (see `status`)
python3 scripts/hf-models.py link                   # recreate the symlinks (e.g. after a fresh clone, if the cache is there)
python3 scripts/hf-models.py adopt                  # one-time: move plain files already in models/ into the cache
                                                    # (SHA-256 checked against the Hub; same disk = instant move)
hf cache ls                                         # what the HF cache holds;  hf cache verify Comfy-Org/MiniMax-H3
```

Sets: `default` = everything the workflows use (bf16 DiT + text encoder, video/audio VAE, 4- and 8-step LoRAs, Q4 GGUF DiT +
text encoder); `q4` = the GGUF pair; `small` = fp8 DiT + nvfp4 text encoder (not used by the shipped workflows); `all`.

| Key | Repo | File → `mini-max-amd-7900xtx/models/<folder>/` | Size |
|---|---|---|---|
| `dit-bf16` | `Comfy-Org/MiniMax-H3` | `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | 40.2 GB |
| `te-bf16` | `Comfy-Org/MiniMax-H3` | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | 51.5 GB |
| `vae-video` | `Comfy-Org/MiniMax-H3` | `vae/minimax_h3_video_vae_fp16.safetensors` | 5.2 GB |
| `vae-audio` | `Comfy-Org/MiniMax-H3` | `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB |
| `lora-4step` | `Comfy-Org/MiniMax-H3` | `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 2.0 GB |
| `lora-8step` | `Comfy-Org/MiniMax-H3` | `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 2.0 GB |
| `dit-q4` | `unsloth/MiniMax-H3-GGUF` | `diffusion_models/minimax_h3_fl2va_pruned-Q4_K.gguf` | 11.4 GB |
| `te-q4` | `unsloth/MiniMax-H3-GGUF` | `text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 18.2 GB |

Manual equivalent for one file (what the script does):

```bash
f=$(hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors)   # prints the cached snapshot path
ln -sfn "$f" mini-max-amd-7900xtx/models/vae/minimax_h3_video_vae_fp16.safetensors            # absolute link into the cache
```

Notes:
- Use **absolute** links (as above): the containers see `models/` at `/opt/ComfyUI/models`, so relative links would break.
- A non-default cache location works through `HF_HUB_CACHE` (or `HF_HOME`); export it before running the script *and*
  `docker compose`, so the mount and the links agree.
- `hf cache rm` / `prune` on these repos breaks the links: re-run `download` (re-fetches) and the links come back.
- `mini-max-amd-7900xtx/scripts/download-models.sh` (inside the container, `--local-dir`) still works but writes plain files into
  `models/`; run `adopt` afterwards to move them into the cache.

## Quick start

```bash
python3 scripts/hf-models.py download        # or `status` if the cache is already populated
cd mini-max-amd-7900xtx && ./scripts/set-amd-render-nodes.sh && docker compose up -d        # AMD   -> http://localhost:8190
cd mini-max-nvidia-3090 && docker compose up -d                               # NVIDIA -> http://localhost:8189
```

Details per stack: [`mini-max-amd-7900xtx/README.md`](mini-max-amd-7900xtx/README.md), [`mini-max-nvidia-3090/README.md`](mini-max-nvidia-3090/README.md).
