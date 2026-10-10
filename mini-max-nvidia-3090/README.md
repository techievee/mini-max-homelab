# mini-max-nvidia-3090: ComfyUI + MiniMax H3 on 4x RTX 3090

The NVIDIA version of [`../mini-max-amd-7900xtx`](../mini-max-amd-7900xtx) (4x RX 7900 XTX). It uses the same ComfyUI v0.39.0 and the same
[Raylight](https://github.com/komikndr/raylight) commit, so one generation is still split across 4 GPUs with Ulysses sequence parallel and FSDP.
It runs on PyTorch 2.12.1 with CUDA 13.2. Four of the host's eight 3090s are passed in, and the AMD cards are not visible to it.
By default those are the four behind the PLX switch, with `NCCL_P2P_LEVEL=PXB`. The four direct-attach cards are an alternative you set in `.env`.

Full test and benchmark report, NVIDIA vs AMD: [`REPORT-2026-10-10.md`](REPORT-2026-10-10.md).
PCIe/P2P tuning session (hardware fixes, ACS/IOMMU kernel args, all bandwidth runs): [`SESSION-2026-10-10-evening-pcie.md`](SESSION-2026-10-10-evening-pcie.md).

## Quick start
```bash
docker compose build
docker compose run --rm --entrypoint torchrun comfyui --nproc_per_node 4 scripts/gpu-check.py   # GPUs, P2P, NCCL check
docker compose up -d                                                                            # http://localhost:8189
```
Models are **shared** with `../mini-max-amd-7900xtx/models` (mounted through `MODELS_DIR` in `.env`), which holds symlinks into the
Hugging Face cache; the compose file also mounts the cache read-only at the same path. Manage them from the repo root with
`python3 scripts/hf-models.py status|download|link|adopt`, see [Models](../README.md#models-hugging-face-cache--symlinks).

The two stacks can run at the same time: the ROCm one (now on port 8190) and this one on 8189 (8-GPU variant: 8191). They use separate GPUs, and they share host RAM and `/dev/shm`.

## Generate from the CLI
```bash
python3 scripts/run-t2v.py --prompt "..." --seconds 15           # 4-step turbo LoRA, 864x480, 4 GPUs (Ulysses 4 + FSDP)
python3 scripts/run-t2v.py --no-lora --steps 20 --seconds 5      # base model
docker exec mini-max-nvidia-3090 python scripts/run-long.py --seconds 30
```
Ready-made UI workflows (Workflows sidebar, `W` key; files in `user/default/workflows/`). All use `XFuser_attention = SAGE_FP16`:
- `h3_t2v_raylight.json`: 4 GPUs, bf16 DiT + FSDP, 4-step turbo LoRA.
- `h3_t2v_raylight_q4.json`: 4 GPUs, Q4 GGUF DiT + Q4 text encoder, FSDP off.
- `h3_t2v_raylight_cuda8.json`: 8 GPUs, 4-step turbo LoRA.
- `h3_t2v_raylight_cuda8_lora8.json`: 8 GPUs, 8-step turbo LoRA at 8 steps, 1344x768. The official template uses 6 steps with this LoRA.
- `h3_t2v_raylight_cuda8_hq.json`: 8 GPUs, no LoRA, 20 steps, 1344x768 (official template sampling).
- `h3_t2v_raylight_cuda8_1080p.json`: 8 GPUs, 1920x1088, 4-step turbo LoRA.

`run-t2v.py` defaults to `--attention SAGE_FP16`. Pass `--attention TORCH_FLASH` for the old backend, and `--lora8` for the 8-step LoRA.

`run-t2v.py` writes the API-format graph it submits to `user/default/api_workflows/`. That folder is kept out of the sidebar so it can't overwrite the UI files.

The UI workflow steps are the same as in `../mini-max-amd-7900xtx/README.md` (RayInitializerAdvanced with `GPU=4`, `ulysses_degree=4`, `FSDP=true`, `TORCH_FLASH`).

Measured 2026-10-08 on the PLX set with `PXB`: pruned bf16 DiT, bf16 text encoder, 4-step LoRA, 864x480.

| Clip | Whole prompt | Note |
|---|---|---|
| 15.08 s / 362 frames, single shot | 385 s | cold start: Ray start + model init (~96 s) + text encoder + VAE decode. Output: `output/video/MiniMax_H3_raylight_00001_.mp4` (h264 + aac) |

## 4-bit (Q4 GGUF) variant
`unsloth/MiniMax-H3-GGUF`: DiT `minimax_h3_fl2va_pruned-Q4_K.gguf` (11.4 GB) + text encoder `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (18.2 GB),
in the shared models dir. Loaded with Raylight's `RayGGUFLoader` + city96 `CLIPLoaderGGUF` (ComfyUI-GGUF is in the image).
```bash
python3 scripts/run-t2v.py --q4 --seconds 15 --prompt "..."     # API dump -> user/default/api_workflows/
```
- The GGUFs have no architecture tag and store the 5-D vision patch-embed flattened; `patches/gguf_minimax_h3.py` (applied at build) handles both.
- **No FSDP with GGUF** (Raylight limitation): each GPU holds the full 11.4 GB DiT; Ulysses still splits the frames 4 ways.
- The Q4 text encoder fits on one 3090, so it no longer streams from system RAM.
- Ampere has no 4-bit math: weights are dequantized to bf16 on the fly, so it saves VRAM and disk, not compute.
- Smoke test 2026-10-10: 864x480, 39 frames (1.6 s), 4-step LoRA: 66 s for the whole prompt, peak ~15 GB per worker GPU / 18 GB on cuda:0.
- Full run 2026-10-10: **1376x768, 362 frames (15.08 s)**, 4-step LoRA, 4 GPUs (FSDP off): **10 min 55 s** for the whole prompt.

  | Stage | Time | GPUs |
  |---|---|---|
  | Q4 text encoder | ~25 s | cuda:0 |
  | Ray start + GGUF load | ~10 s | 4 |
  | Sampling, 4 steps | **8 min 05 s** (~2 min/step) | 4 at 100% |
  | VAE decode + mux | ~2 min 10 s | cuda:0 |

  Peak VRAM was 14.4 GB per GPU. Host RAM went from 85 to 153 GB (container 81 GB, mostly the GGUF weights mmap'd by the 4 workers). Swap stayed at 0 and `/dev/shm` at 4.8 GB, so nothing spilled.

## All 8 GPUs (`cuda8`)
Second compose service `comfyui-cuda8` (profile `cuda8`, container `mini-max-nvidia-3090-cuda8`, **port 8191**) gets all eight 3090s.
It overlaps the 4-GPU service on the PLX cards, so run one at a time:
```bash
docker compose stop comfyui && docker compose --profile cuda8 up -d comfyui-cuda8      # http://localhost:8191
docker compose stop comfyui-cuda8 && docker compose up -d comfyui                       # back to 4 GPUs
```
Workflow: `h3_t2v_raylight_cuda8.json`. Settings: bf16 DiT, FSDP on, `GPU=8`, `ulysses_degree=8`. The 56 attention heads split as 7 per GPU, and each GPU holds ~5 GB of weights.
The GGUF Q4 DiT can't use FSDP, so the 8-GPU workflow uses bf16.

Rank grouping: `CUDA_VISIBLE_DEVICES=0,5,6,7,1,2,3,4`. Ray hands out GPUs to the Raylight ranks in that order.

| ranks | cards | link to next rank |
|---|---|---|
| 0-1 | 01:00.0, 41:00.0 | NVLink |
| 1-2 | 41:00.0 → 42:00.0 | PHB (same CPU root complex 40) |
| 2-3 | 42:00.0, 61:00.0 | NVLink |
| 3-4 | 61:00.0 → 30:00.0 | NODE (host shared memory) |
| 4-7 | 30, 33, 34, 37:00.0 | PLX switch (PIX/PXB) |
| 7-0 | 37:00.0 → 01:00.0 | NODE (host shared memory) |

`NCCL_P2P_LEVEL=PXB` (override with `NCCL_P2P_LEVEL_CUDA8` in `.env`). Tested 2026-10-10 with `torchrun --nproc_per_node 8 scripts/gpu-check.py`:
- `PHB`: **hangs**. P2P on the 41↔42 hop through the CPU root complex locks up, the same as `SYS`.
- `PXB`: works, all-reduce bus bandwidth **4.0 GB/s**. P2P on 01→41, 42→61 and 30→33→34→37; shared memory on 41→42, 61→30 and 37→01.

Measured 2026-10-10: bf16 DiT, FSDP across 8 GPUs, Ulysses 8, `PXB`, **1376x768, 362 frames (15.08 s)**, 4-step LoRA. Whole prompt **10 min 46 s**.
Output: `output/video/MiniMax_H3_raylight_cuda8_00001_.mp4`.

| Stage | Time | GPUs |
|---|---|---|
| bf16 text encoder | ~2 min 36 s | cuda:0 (peaked at **23.3 GB** of 24) |
| Ray start, comm test, FSDP shard load | ~45 s | 8 |
| Sampling, 4 steps | **5 min 06 s** (~76 s/step) | all 8 at ~100% |
| VAE decode + mux | ~2 min 10 s | cuda:0 |

Peak VRAM during sampling was ~14 GB per GPU and ~19 GB after load. Host RAM peaked at 156 GB, swap stayed at 0, and `/dev/shm` peaked at 9.6 GB.
Sampling ran 1.6x faster than the 4-GPU Q4 run at the same size (8 min 05 s), not 2x. Likely causes (not measured): the shared-memory hops through the PLX uplink, plus FSDP all-gather, which the Q4 run did not need.

**1080p** workflow: `h3_t2v_raylight_cuda8_1080p.json` (1920x1088, 362 frames). Test 2026-10-10 at **1920x1088, 124 frames (5.2 s)**, 8 GPUs: **5 min 16 s** total.
- Text encoder took ~33 s; it was still cached from the previous run.
- Sampling: **3 min 00 s**, all 8 GPUs at 100%, peak **15.5 GB** per GPU.
- VAE decode: ~1 min 35 s on cuda:0.
- Host RAM peaked at 157 GB, swap stayed at 0.
- Quality is coherent at 1080p with no tiling or duplicated subjects. The usual weak spots remain: tangled limbs where moving subjects overlap, and garbled signage text.

### Native config (no LoRA, 20 steps) and SageAttention
Workflows: `h3_t2v_raylight_cuda8_hq.json` (official template sampling: base model, BasicGuider, res_multistep/simple, 20 steps, 1344x768) and
`h3_t2v_raylight_cuda8_hq_sage.json` (same, `XFuser_attention = SAGE_FP16`).
SageAttention 2.2 is built into the image for sm_86 and sm_89. On Ampere `SAGE_FP16` is the best kernel (INT8 QK, FP16 PV); the FP8 variants need sm_89+.

Measured 2026-10-10: 1344x768, 362 frames, same prompt and seed, 8 GPUs, FSDP, Ulysses 8.

| Attention | Sampling (20 steps) | Per step | Whole prompt | Peak VRAM (sampling) |
|---|---|---|---|---|
| `TORCH_FLASH` | 22 min 34 s | ~68 s | 26 min 14 s | 16.6 GB |
| **`SAGE_FP16`** | **16 min 34 s** | **~50 s** | **22 min 04 s** (incl. a ~2 min model reload) | 16.6 GB |

- **Sage is 1.36x faster on sampling.** Kernel microbenchmark: 11.6 ms vs 18.7 ms (1.6x), cosine similarity 0.99993.
- **Quality:** same shots, timeline and on-screen text, with no visible artefacts.
- Frame SSIM vs Flash is 0.84 (PSNR 21.8 dB), because 20 steps amplify tiny numeric differences into slightly different camera position and motion.
- The 4-step turbo LoRA (5 min sampling) loses most of the prompt's timeline, e.g. the 10–15 s end card; the base model renders it correctly.

## GPU selection and P2P level
**Do not use `NCCL_P2P_LEVEL=SYS` on this host.** It forces PCIe P2P through the CPU root complex, and the NCCL all-reduce check hung on every try.
Pick the level that matches the GPU set:

| Set (`.env`) | PCI | Path between GPUs | `NCCL_P2P_LEVEL` | `gpu-check.py` all-reduce |
|---|---|---|---|---|
| **PLX (default)** | 30, 33, 34, 37:00.0 (host idx 1-4) | PLX/PEX switch on root port 20:03.1 (topo `PXB`/`PIX`) | `PXB` | **6.4 GB/s**, P2P on all pairs |
| Direct-attach | 01, 41, 42, 61:00.0 (host idx 0, 5, 6, 7) | separate CPU root ports, NVLink pairs 01↔41 and 42↔61 | `NVL` | 11.9 GB/s, NVLink P2P + SHM for the rest |
| Direct-attach | same | same | `SYS` | **hangs** |

PLX set inside the container (`CUDA_DEVICE_ORDER=PCI_BUS_ID`):

| host idx | PCI | in container | link |
|---|---|---|---|
| 1 | 30:00.0 | cuda:0 | x16 |
| 2 | 33:00.0 | cuda:1 | **x8** |
| 3 | 34:00.0 | cuda:2 | x16 |
| 4 | 37:00.0 | cuda:3 | x16 |

`p2p-matrix.py` on the PLX set: **25.5 GB/s** between the x16 cards and 13.2 GB/s to and from cuda:1, which is limited by its x8 link.
H2D/D2H is about 24-25 GB/s. Peer traffic is switched inside the PLX, unlike on the ROCm host, where ACS redirect capped it at 14 GB/s.

To switch to the direct-attach set, uncomment `GPU_A`..`GPU_D` and `NCCL_P2P_LEVEL=NVL` in `.env`, then run `docker compose up -d --force-recreate`.
GPUs are selected by **UUID**, so a change in nvidia-smi index order can't pull in the wrong card. After a hardware change, re-check with:
```bash
nvidia-smi -L; nvidia-smi topo -m
for b in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do readlink -f /sys/bus/pci/devices/${b#0000}; done
```

## Notes vs the ROCm build
- There are no `/dev/kfd` or render-node mappings, no `amdsmi` shim and no MIOpen cache. GPU access goes through the NVIDIA Container Toolkit
  (`deploy.resources.reservations.devices`), and Ray finds the GPUs through NVML.
- The base image keeps torch in the system Python, which has a PEP 668 marker, so `PIP_BREAK_SYSTEM_PACKAGES=1` is set. Raylight imports `torchaudio`,
  but there is no 2.12 release, so the image pins `torchaudio==2.11.0+cu132` with `--no-deps`. The ROCm image uses the same torch 2.12 + torchaudio 2.11 pair.
- The `int8_convrot` checkpoints need CUDA 13 kernels. This image has CUDA 13.2, but they are untested on Ampere (sm_86), so the download script still skips them.
- Each 3090 has 24 GB, the same as a 7900 XTX, so the memory notes for the 4-GPU split carry over: bf16 DiT + FSDP, and `FSDP_CPU_OFFLOAD` if memory runs out.
