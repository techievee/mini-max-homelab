# mini-max: ComfyUI + MiniMax H3 on 4x RX 7900 XTX

ComfyUI v0.39.0 (native MiniMax H3), with [Raylight](https://github.com/komikndr/raylight) to split one generation across all
four 7900 XTX cards. Raylight uses Ulysses sequence parallel for activations and FSDP for weights. Runs on ROCm 7.14 / PyTorch 2.12.
Only the AMD render nodes are passed into the container, so the RTX 3090s are not visible to it.

## Quick start
```bash
docker compose build
docker compose run --rm --entrypoint torchrun comfyui --nproc_per_node 4 scripts/gpu-check.py   # GPUs, P2P, RCCL check
python3 ../scripts/hf-models.py download                                                         # host: ~131 GB into the HF cache + symlinks
docker compose up -d                                                                            # http://localhost:8190
```

Models live in the Hugging Face cache; `models/` holds symlinks into it and the compose file mounts the cache read-only at the
same path. See [Models](../README.md#models-hugging-face-cache--symlinks) in the repo README (`status` / `download` / `link` / `adopt`).

Legacy in-container downloader (writes plain files into `models/`; run `../scripts/hf-models.py adopt` afterwards),
`docker compose run --rm --entrypoint bash comfyui scripts/download-models.sh` flags:
- `--small`: fp8_scaled DiT + nvfp4 text encoder, ~45 GB.
- `--ref2va`: also fetch the Ref2VA checkpoint.
- `--dry-run`: print the file list only.

The `int8_convrot` checkpoints are skipped. Upstream says they need CUDA 13 kernels.

## Tuned config and benchmark (2026-10-10)
PCIe/P2P session (riser + slot fixes, `iommu=pt pci=disable_acs_redir=pci:1000:c010`, RCCL 7 → ~25 GB/s): [`../mini-max-nvidia-3090/SESSION-2026-10-10-evening-pcie.md`](../mini-max-nvidia-3090/SESSION-2026-10-10-evening-pcie.md).

Full report: [`../mini-max-nvidia-3090/REPORT-2026-10-10.md`](../mini-max-nvidia-3090/REPORT-2026-10-10.md), which covers all tests, attention backends, PCIe findings and the comparison with the 3090s.

A second, tuned service runs on the same 4 GPUs, so stop `comfyui` first:
```bash
docker compose stop comfyui && docker compose --profile bench up -d --build comfyui-bench   # http://localhost:8192
python3 scripts/bench.py screen          # benchmark suites; every output is checked (black-frame guard)
```
The image adds ComfyUI-GGUF plus `patches/gguf_minimax_h3.py`. `scripts/h3_workflow.py` gains `attention`, `ring`, `lora_name`, GGUF (`Q4_UNET`/`Q4_CLIP`) and `sla`.

Measured at 1344x768, 362 frames (15 s), 4-step LoRA, 4x 7900 XTX:

| Config | Total | Sampling | Note |
|---|---|---|---|
| bf16 DiT + FSDP, bf16 TE, AOTriton Flash (previous) | 11:06 | 7:20 | |
| **Q4 GGUF DiT + Q4 TE, Ulysses 4, AOTriton Flash** | **8:50** | 6:33 | **recommended** |
| same + `RayMiniMaxH3SLA` 75 % sparse attention | 6:47 | 4:34 | fastest; check shot timing on multi-shot prompts |
| Q4, 8-step LoRA at 8 steps | 15:20 | 13:03 | follows the timeline and the end card |
| Q4, no LoRA, 20 steps | 33:45 | 31:29 | best prompt-following |

- **UI workflow for the recommended config:** `user/default/workflows/h3_t2v_rocm_q4.json` (sidebar on :8192). It uses Q4 GGUF DiT + Q4 text encoder,
  FSDP off, Ulysses 4, `TORCH_FLASH`, 1344x768, 362 frames, 4-step LoRA, saving to `video/MiniMax_H3_rocm_q4`. It needs the `comfyui-bench` container.
- **Attention on RDNA3:** keep `TORCH_FLASH` (AOTriton, needs `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`). SageAttention's Triton path is 3x slower, and flash-attn's AMD path doesn't build in this image.
- Ulysses 4 beats Ulysses 2 x Ring 2.
- **Restart the container whenever you switch between bf16+FSDP and Q4.** Mixing them corrupts the next run: Q4 after FSDP comes out black, FSDP after Q4 comes out as pixel noise. `bench.py` restarts the container automatically and checks every output.
- **PCIe:** the PLX switch's uplink to root port 00:03.1 trains at **Gen4 x8**, and so does card 08:00.0. Both are capable of x16. Fix the seating, slot or lane-sharing, and disable ACS (see below).

## Generate from the CLI (verified)
```bash
python3 scripts/run-t2v.py --prompt "..." --seconds 5            # 4-step turbo LoRA, 864x480, 4 GPUs (Ulysses 4 + FSDP)
python3 scripts/run-t2v.py --no-lora --steps 20 --seconds 5      # base model
```
The script submits a Raylight API workflow, waits for it, and prints the mp4 path under `output/video/`.
It also saves that workflow to `user/default/workflows/api_h3_t2v_raylight.json`; drag it into the UI to get the same graph.

Measured 2026-10-08 (pruned bf16 DiT + bf16 text encoder, 4-step LoRA, 864x480, 4x 7900 XTX):

| Clip | Sampling | Whole prompt | Note |
|---|---|---|---|
| 2.3 s / 56 frames | 46 s (~10 s/step) | 220 s | cold: Ray start + 49 GB text-encoder load |
| 5.2 s / 124 frames | 56 s (~12 s/step) | 178 s | warm |

During sampling all 4 GPUs run at 100%, using ~55% VRAM each. Most of the remaining time goes to the 32B text encoder on cuda:0 and to VAE decode.

## Long clips (30 s+)
H3 is trained on roughly 5-15 s, so longer clips are chained. `scripts/run-long.py`:
1. Generates ~10.8 s segments (260 frames).
2. Anchors each new segment on the previous one's last 22 frames *and audio* (`MiniMaxH3AddGuide` at frame 0).
3. Drops the overlap and stitches everything with ffmpeg.

Run it inside the container, since Fedora's ffmpeg lacks libx264:
```bash
docker exec mini-max-amd-7900xtx python scripts/run-long.py --seconds 30 --name my_clip [--story story.json]
#  story.json = {"style": "shared look + character description", "beats": ["seg 1 action", "seg 2", "seg 3"]}
#  --existing output/video/my_clip_seg1_00001_.mp4   resume after a failure
```
Measured: 30.7 s / 737 frames in ~16 min (3 segments at ~5 min each, plus a Ray restart between segments).
Example output: `output/video/coastal_ride_31s.mp4`.
The script also aborts if a segment comes out black.

### Fixes baked into this setup
- **Prompt ignored, or black video.** ComfyUI ran the 32B text encoder in fp16 on RDNA3, and it overflows on some prompts.
  Fixed with `command: ["--bf16-text-enc"]` in compose.
- **OOM: "Tried to allocate 7.17 GiB" on GPU0** when encoding continuation frames. After a long segment, Raylight's Ray worker on GPU0 keeps ~16 GB.
  `run-long.py` shuts down the Ray cluster (`RayKill`) between segments and runs the text encoder on CPU (`--clip-device cpu`).
- **Ray saw 0 AMD GPUs.** Ray's AMD detection needs the `amdsmi` Python bindings. The Dockerfile installs `amdsmi==7.0.2` and links
  `/opt/rocm/lib/libamd_smi.so` to the TheRock ROCm lib inside the venv, with `LD_LIBRARY_PATH` set.
  The earlier `~/comfy-rocm` setup patched `ray.init(num_gpus=...)` instead.
- **Ray `OutOfDiskError`.** The root disk is >95% full, so `RAY_local_fs_capacity_threshold=0.995` is set and the object store is raised to 32 GB (in /dev/shm).
- **`Too many open files`** while staging the 49 GB text encoder. Fixed with `nofile` ulimit 524288.

## 4-GPU workflow (Raylight)
1. Start from **Templates → MiniMax H3: Text to Video**. Keep the text encoder, `EmptyMiniMaxH3LatentAV`, and the audio/video save nodes.
2. Add **RayInitializerAdvanced** with these settings:
   - `GPU=4`, `ulysses_degree=4`, `ring_degree=1`, `cfg_degree=1`
   - `FSDP=true`
   - `XFuser_attention=TORCH_FLASH`
   - `use_mmap=true`
3. Replace the loaders:
   - Use **RayUNETLoader** with `minimax_h3_fl2va_pruned_bf16`.
   - Optionally add **RayLoraLoader** with the 4-step turbo LoRA. With the LoRA, use steps=4 and cfg=1.
   - Then add **RayMiniMaxH3SigmaShift** (video 12 / audio 3), in place of the stock SigmaShift and model loader.
4. Replace the sampler with **XFuserKSamplerAdvanced**.
5. VAE decode: **RayVAEDecodeDistributed** for video, or the stock decoder.
6. Save the result as `user/default/workflows/h3_raylight_4gpu.json`.

Alternative split: if cfg > 1, set `ulysses_degree=2` and `cfg_degree=2`.

If memory runs out:
- Turn on `FSDP_CPU_OFFLOAD` (377 GB of RAM is available).
- Or set `sync_ulysses=true`.

## GPU mapping
| PCI | render node | in container |
|---|---|---|
| 08:00.0 | renderD130 | cuda:0 (slot trains at **Gen4 x8**) |
| 0e:00.0 | renderD131 | cuda:1 |
| 13:00.0 | renderD132 | cuda:2 |
| 16:00.0 | renderD133 | cuda:3 |

Render node numbers can change after a kernel or hardware change. Re-check with `ls -l /dev/dri/by-path` and edit `docker-compose.yml`.

## P2P / RCCL bandwidth (host tuning, needs root)
Measured on 2026-10-08. Run the scripts with `docker compose run --rm --entrypoint python comfyui scripts/<name>.py`.

| Test | Result |
|---|---|
| `p2p-matrix.py`: one copy between any GPU pair, or host to/from device | **13.8 GB/s** each |
| `p2p-concurrent.py`: 0->1 + 2->3 at once, bidirectional, or 4-way ring | **13.8 GB/s total**, never more |
| `gpu-check.py` (torchrun x4): RCCL all-reduce, ROCm 7.14 | 3.5 GB/s bus bandwidth (3.0 with `NCCL_P2P_DISABLE=1`) |
| Same RCCL all-reduce, ROCm 10.0 (`rocm/vllm` rocm10 image) | 3.5 GB/s. Identical, so this is not a ROCm version issue |

Every peer transfer goes through one shared link of about 14 GB/s. That matches the CPU root port 00:03.1, which runs at Gen4 x8.
So even though the GPUs sit on a common PLX/PEX switch, their traffic is not switched locally. PCIe **ACS** (Access Control Services) on the switch's
downstream ports redirects peer requests upstream to the root complex, and IOMMU groups do not change this.
Disabling ACS should let pairs talk at x16 speed (~25 GB/s) and in parallel:

```bash
sudo lspci -vvv | grep -E '^[0-9a-f]|ACSCtl' | grep -B1 ACSCtl     # SrcValid+ ReqRedir+ CmpltRedir+ = redirect on
# Option 1: BIOS setting "ACS" -> Disabled (or "PCIe P2P"/"Above 4G" options on some boards).
# Option 2: per boot, clear ACS control on the switch ports above the GPUs:
for b in 03:00.0 03:04.0 03:08.0 05:10.0 0a:10.0 10:00.0 10:10.0; do sudo setpci -s $b ECAP_ACS+0x6.w=0000; done
```
After that, re-run `p2p-concurrent.py`. "0->1 + 2->3" should go well above 14 GB/s.
Also, GPU0 (08:00.0) has a link that trains at Gen4 x8 under load; reseat the card or check its riser.
