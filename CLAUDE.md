# CLAUDE.md — mini-max-homelab

MiniMax H3 (text → video + audio) in ComfyUI + Raylight, two Docker Compose stacks on one host:

| Folder | GPUs | Containers (service → name, port) |
|---|---|---|
| `mini-max-amd-7900xtx/` | 4× RX 7900 XTX (ROCm 7.14) behind one PEX88096 switch | `comfyui` → `mini-max-amd-7900xtx` :8190, `comfyui-bench` (profile `bench`, has ComfyUI-GGUF) → `mini-max-amd-7900xtx-bench` :8192 |
| `mini-max-nvidia-3090/` | 4× RTX 3090 on a PEX88096 (default) / 4 direct-attach / all 8 (CUDA 13.2) | `comfyui` → `mini-max-nvidia-3090` :8189, `comfyui-cuda8` (profile `cuda8`) → `mini-max-nvidia-3090-cuda8` :8191 |

Read [`README.md`](README.md) (topology, P2P, models) and `mini-max-nvidia-3090/REPORT-2026-10-10.md` (all benchmarks and fixes) before
changing GPU, NCCL or attention settings.

## Ground rules

- **Verify every generation before starting the next test:** an mp4 can be black (NaN) or pixel noise. Check mean luma and
  KB/frame (`mini-max-amd-7900xtx/scripts/bench.py: check_output` — good = luma ≥ 25 and 3–100 KB/frame) and look at a frame.
- Report measured numbers (timings per phase, actual power), label estimates.
- Don't stop a container someone is running a job in; ask before destructive cleanup (`docker image prune`, deleting outputs).
- sudo is not passwordless: give the exact command / script instead of running it.
- The NVIDIA and AMD groups have separate PSUs, so they can benchmark in parallel.
- Never commit model weights, `output/`, `cache/`, `input/`, `.env`, ComfyUI logs or ComfyUI-Manager state (see `.gitignore`).

## Models

Weights live in the Hugging Face cache; `mini-max-amd-7900xtx/models/*` are absolute symlinks into it, shared by both stacks
(`MODELS_DIR=../mini-max-amd-7900xtx/models`). Both compose files mount the HF hub cache read-only at the same absolute path.
Manage with `python3 scripts/hf-models.py status|download|link|adopt`. Run compose from the real folders (a symlinked
parent path changes what `../` resolves to; a missing bind source makes Docker mount an empty, root-owned dir).

## GPU / P2P rules (details in README)

- `NCCL_P2P_LEVEL`: **PXB** for the PLX 3090s and the AMD cards, **NVL** for the direct-attach 3090s. Never `PHB`/`SYS`
  (P2P through the CPU root complexes hangs here); not `PIX` on AMD (turns P2P off: 3.6 vs 25.6 GB/s).
- Host needs kernel args `iommu=pt pci=disable_acs_redir=pci:1000:c010`, otherwise collectives are ~4× slower.
- Cards are selected by UUID / PCI path, not index. AMD render nodes shift after BIOS changes:
  run `mini-max-amd-7900xtx/scripts/set-amd-render-nodes.sh` (writes `.env`) before `docker compose up`.
- 8-GPU rank order (`cuda8`): `CUDA_VISIBLE_DEVICES=6,0,1,7,2,3,4,5` (NVLink pairs adjacent, PLX cards last).
- Slots 01/02 (root complex 00) log PCIe link errors and once hard-reset the host under load: avoid heavy jobs on them
  until the slots run Gen3 / risers are fixed. Their BAR1 is 256 MiB (VBIOS, not BIOS).

## Known workarounds

- Raylight/Ray crashes (`Cannot find an actor handle`) when the Ray config changes in place (GPU count, `GPU_SELECT`, or after a
  failed job): restart the container between such switches.
- ROCm only: mixing bf16+FSDP and Q4 GGUF runs in one container corrupts later runs (Q4 after FSDP → black; FSDP after Q4 →
  noise). Restart on every bf16 ↔ GGUF switch (`bench.py` does it). After a bf16+FSDP job, rank 0 can stay at 100 % busy with
  spinning Ray workers: restart the container.
- ComfyUI caches identical prompts (re-run returns in ~3 s with no work): change seed or a Ray input to force a run.
- Text encoder must be bf16 on ROCm (`--bf16-text-enc`); fp16 gives black / prompt-ignoring video.
- GGUF can't use FSDP in Raylight; unsloth MiniMax GGUFs need `patches/gguf_minimax_h3.py` (baked into the images).
- Text encoder + VAE decode always run on cuda:0 (no distributed VAE decode for H3).
- Attention: NVIDIA best = `SAGE_FP16` (default in the NVIDIA workflows); AMD best = `TORCH_FLASH` (AOTriton).
- Long clips: `scripts/run-long.py` chains 10.8 s segments with tail-frame + audio guidance (~16 min for 30 s on AMD);
  kill Ray between segments or cuda:0 runs out of memory on the guide VAE encode.
- 3090 clocks: a boot service may cap 250 W and lock 1350 MHz (~25 % slower); `mini-max-nvidia-3090/scripts/gpu-power.sh
  default|capped|status` (sudo) switches. Check before benchmarking.

## Quick checks

```bash
python3 scripts/hf-models.py status
cd mini-max-amd-7900xtx && ./scripts/set-amd-render-nodes.sh && docker compose up -d && python3 scripts/run-t2v.py --host http://localhost:8190 --seconds 2
cd mini-max-nvidia-3090 && docker compose up -d && python3 scripts/run-t2v.py --host http://localhost:8189 --seconds 2
```
`run-t2v.py` rewrites the API workflow it submits (AMD: `user/default/workflows/api_h3_t2v_raylight.json`, tracked) —
`git restore` it after test runs.
