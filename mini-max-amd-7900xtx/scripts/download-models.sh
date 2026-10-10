#!/usr/bin/env bash
# Download MiniMax H3 files from Comfy-Org/MiniMax-H3 into the ComfyUI models folder.
# Run inside the container:  docker compose run --rm --entrypoint bash comfyui scripts/download-models.sh [flags]
#
# Default set is ROCm-safe (no int8_convrot: those need CUDA 13 kernels):
#   FL2VA pruned bf16 (40 GB, ~10 GB/GPU with Raylight FSDP), Qwen3-VL text encoder bf16 (51.5 GB),
#   video + audio VAE, 4-step turbo LoRA.                                       Total ~100 GB
#
# Flags:
#   --small    FL2VA pruned fp8_scaled (21 GB) + nvfp4 text encoder (15.7 GB) instead   (~45 GB total)
#   --ref2va   also fetch the Ref2VA checkpoint (same precision as the FL2VA choice)
#   --dry-run  print the file list only
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/opt/ComfyUI/models}"
REPO="Comfy-Org/MiniMax-H3"

dit="pruned_bf16"
te="bf16"
ref2va=0
dry=0
for arg in "$@"; do
  case "$arg" in
    --small)   dit="pruned_fp8_scaled"; te="nvfp4_awq" ;;
    --ref2va)  ref2va=1 ;;
    --dry-run) dry=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

files=(
  "diffusion_models/minimax_h3_fl2va_${dit}.safetensors"
  "text_encoders/qwen3vl_32b_minimax_h3_${te}.safetensors"
  "vae/minimax_h3_video_vae_fp16.safetensors"
  "vae/minimax_h3_audio_vae_fp32.safetensors"
  "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
)
if [[ $ref2va == 1 ]]; then
  files+=(
    "diffusion_models/minimax_h3_ref2va_${dit}.safetensors"
    "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
  )
fi

printf '%s\n' "${files[@]}"
[[ $dry == 1 ]] && exit 0

# Repo layout already matches ComfyUI's models/<folder>/ layout, so download straight in.
hf download "$REPO" "${files[@]}" --local-dir "$MODELS_DIR"
rm -rf "$MODELS_DIR/.cache"
echo "Done. Files are in $MODELS_DIR"
