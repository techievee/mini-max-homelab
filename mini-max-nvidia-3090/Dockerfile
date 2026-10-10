# ComfyUI + Raylight (multi-GPU) for MiniMax H3 on NVIDIA Ampere (RTX 3090 / sm_86)
FROM pytorch/pytorch:2.12.1-cuda13.2-cudnn9-devel

ARG COMFYUI_REF=v0.39.0
ARG RAYLIGHT_REF=74f5defa8a4d0e6e62cf0ea5e717ea6b4305bb13
ARG MANAGER_REF=main
ARG GGUF_REF=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
ARG SAGE_REF=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5

# torch lives in this image's system Python (PEP 668 marker); it's a single-purpose container, so let pip write there.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_CONSTRAINT=/opt/torch-constraints.txt \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# The base image has no torchaudio, and Raylight imports it. 2.11 is the last torchaudio release (same pairing as the
# ROCm image: torch 2.12 + torchaudio 2.11). --no-deps keeps it from replacing torch; the CUDA build has to be cu132.
# Then freeze the CUDA torch stack so no requirements file can swap it for another build.
RUN PIP_CONSTRAINT= pip install --no-deps --index-url https://download.pytorch.org/whl/test/cu132 "torchaudio==2.11.0+cu132" \
 && python -c "import torchaudio; print('torchaudio', torchaudio.__version__)" \
 && PIP_CONSTRAINT= pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton)==' > /opt/torch-constraints.txt \
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
 && pip install "huggingface_hub[cli]" hf_transfer nvidia-ml-py

# city96 ComfyUI-GGUF: CLIPLoaderGGUF for the 4-bit text encoder, and its presence turns on Raylight's RayGGUFLoader.
# unsloth's MiniMax H3 GGUFs have no architecture tag, so both loaders get a small patch (patches/gguf_minimax_h3.py).
COPY patches/ /opt/patches/
RUN git clone https://github.com/city96/ComfyUI-GGUF.git custom_nodes/ComfyUI-GGUF \
 && git -C custom_nodes/ComfyUI-GGUF checkout "${GGUF_REF}" \
 && pip install -r custom_nodes/ComfyUI-GGUF/requirements.txt \
 && python /opt/patches/gguf_minimax_h3.py custom_nodes/ComfyUI-GGUF/loader.py \
      custom_nodes/raylight/src/raylight/expansion/comfyui_gguf/loader.py

# SageAttention 2.2 (thu-ml): INT8 QK + FP16 PV CUDA kernel = Raylight XFuser_attention "SAGE_FP16" (best on Ampere).
# Built for sm_86 (RTX 3090) and sm_89 (RTX 4090 also gets the FP8-PV kernel "SAGE_FP8"); no GPU needed at build time.
RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/sage \
 && git -C /tmp/sage checkout "${SAGE_REF}" \
 && cd /tmp/sage && TORCH_CUDA_ARCH_LIST="8.6;8.9" EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=16 \
      pip install --no-build-isolation . \
 && cd / && rm -rf /tmp/sage \
 && python -c "import sageattention; print('sageattention', sageattention.__file__)"

# Fail the build early if anything replaced the CUDA torch build.
RUN python -c "import torch; assert torch.version.cuda, torch.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
 && python -c "import pynvml, ray; print('pynvml + ray import ok')"

# The base image only has root; add the host user (uid/gid 1000) so bind-mounted files stay yours.
RUN (getent group 1000 || groupadd -g 1000 comfy) \
 && (getent passwd 1000 || useradd -m -u 1000 -g 1000 -s /bin/bash comfy) \
 && mkdir -p /home/comfy && chown 1000:1000 /home/comfy

COPY --chown=1000:1000 scripts/ /opt/ComfyUI/scripts/

RUN chown -R 1000:1000 /opt/ComfyUI
USER 1000:1000
ENV HOME=/home/comfy

EXPOSE 8188
ENTRYPOINT ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--use-pytorch-cross-attention"]
