# ComfyUI + Raylight (multi-GPU) for MiniMax H3 on AMD RDNA3 (RX 7900 XTX / gfx1100)
FROM rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0

ARG COMFYUI_REF=v0.39.0
ARG RAYLIGHT_REF=74f5defa8a4d0e6e62cf0ea5e717ea6b4305bb13
ARG MANAGER_REF=main
ARG GGUF_REF=6ea2651e7df66d7585f6ffee804b20e92fb38b8a

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_CONSTRAINT=/opt/torch-constraints.txt \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Freeze the ROCm torch stack so no requirements file can swap it for a CUDA/CPU build.
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton)==' > /opt/torch-constraints.txt \
 && cat /opt/torch-constraints.txt

WORKDIR /opt/ComfyUI

RUN git clone https://github.com/Comfy-Org/ComfyUI.git . \
 && git checkout "${COMFYUI_REF}" \
 && grep -viE '^(torch|torchvision|torchaudio)([<>=~ ]|$)' requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt

RUN git clone https://github.com/komikndr/raylight.git custom_nodes/raylight \
 && git -C custom_nodes/raylight checkout "${RAYLIGHT_REF}" \
 && grep -viE '^(torch|torchvision|torchaudio)([<>=~ ]|$)' custom_nodes/raylight/requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt \
 && git clone https://github.com/Comfy-Org/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager \
 && git -C custom_nodes/ComfyUI-Manager checkout "${MANAGER_REF}" \
 && grep -viE '^(torch|torchvision|torchaudio)([<>=~ ]|$)' custom_nodes/ComfyUI-Manager/requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt \
 && pip install "huggingface_hub[cli]" hf_transfer amdsmi==7.0.2

# Ray detects AMD GPUs via the `amdsmi` Python bindings, which look for /opt/rocm/lib/libamd_smi.so.
# This TheRock-based image ships ROCm inside the venv instead, so point the bindings at it.
# Without this, Ray reports 0 GPUs and Raylight workers never get scheduled.
ENV ROCM_SDK_LIB=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
ENV LD_LIBRARY_PATH=${ROCM_SDK_LIB}:${ROCM_SDK_LIB}/rocm_sysdeps/lib
RUN mkdir -p /opt/rocm/lib \
 && ln -sf "$(ls ${ROCM_SDK_LIB}/libamd_smi.so.* | head -1)" /opt/rocm/lib/libamd_smi.so

# city96 ComfyUI-GGUF: CLIPLoaderGGUF for the Q4 text encoder, and its presence turns on Raylight's RayGGUFLoader.
# unsloth's MiniMax H3 GGUFs have no architecture tag, so both loaders get a small patch (patches/gguf_minimax_h3.py,
# shared with ../mini-max-nvidia-3090).
COPY patches/ /opt/patches/
RUN git clone https://github.com/city96/ComfyUI-GGUF.git custom_nodes/ComfyUI-GGUF \
 && git -C custom_nodes/ComfyUI-GGUF checkout "${GGUF_REF}" \
 && pip install -r custom_nodes/ComfyUI-GGUF/requirements.txt \
 && python /opt/patches/gguf_minimax_h3.py custom_nodes/ComfyUI-GGUF/loader.py \
      custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/loader.py

# Fail the build early if anything replaced the ROCm torch build.
RUN python -c "import torch; assert torch.version.hip, torch.__version__; print('torch', torch.__version__, 'hip', torch.version.hip)" \
 && python -c "import amdsmi, ray; print('amdsmi + ray import ok')"

COPY --chown=1000:1000 scripts/ /opt/ComfyUI/scripts/

# Run as the host user (uid/gid 1000 = 'ubuntu' in the base image) so bind-mounted files stay yours.
RUN chown -R 1000:1000 /opt/ComfyUI
USER 1000:1000
ENV HOME=/home/ubuntu

EXPOSE 8190
ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--port", "8190", "--use-pytorch-cross-attention"]
