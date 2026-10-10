"""Shared helpers: build a MiniMax H3 + Raylight API workflow and run it on ComfyUI."""
import json, time, urllib.error, urllib.request, uuid

TURBO_LORA = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
TURBO_LORA_8STEP = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"  # official template pairs it with 6 steps
# 4-bit GGUFs from unsloth/MiniMax-H3-GGUF (loaders patched by patches/gguf_minimax_h3.py; needs the bench image).
Q4_UNET = "minimax_h3_fl2va_pruned-Q4_K.gguf"
Q4_CLIP = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"


def snap_length(seconds):
    """Frame count at 24 fps on the model's 17k+5 grid (same formula as the official template)."""
    f = max(5, round(seconds * 24))
    return f + (5 - f % 17) % 17


def build(prompt, width=864, height=480, length=124, steps=4, seed=42, gpus=4, ulysses=None,
          fsdp=True, lora=True, unet="minimax_h3_fl2va_pruned_bf16.safetensors",
          clip="qwen3vl_32b_minimax_h3_bf16.safetensors", clip_device="default",
          guide_video=None, guide_frames_audio=True, prefix="video/MiniMax_H3_raylight", ray_init_salt=0,
          attention="TORCH_FLASH", ring=1, lora_name=TURBO_LORA, sla=None):
    """guide_video: filename in ComfyUI's input/ dir whose frames (+audio) are anchored at frame 0.
    attention: XFuser backend. On RDNA3 TORCH_FLASH (AOTriton) is the fastest dense kernel; Sage/flash-attn Triton are slower.
    ring: ring_degree (ulysses defaults to gpus // ring).
    unet/clip ending in .gguf: Raylight RayGGUFLoader / CLIPLoaderGGUF (FSDP is forced off; GGUF can't be sharded).
    sla: sparsity ratio for RayMiniMaxH3SLA block-sparse attention (e.g. 0.75 / 0.9), None = dense."""
    wf = {
        "ray": {"class_type": "RayInitializerAdvanced", "inputs": {
            "ray_cluster_address": "local", "ray_cluster_namespace": "default",
            "GPU": gpus, "GPU_SELECT": "", "ulysses_degree": ulysses or gpus // ring, "ring_degree": ring,
            "clear_vram_after_sampling": True, "cfg_degree": 1, "dp_degree": 1, "sync_ulysses": False,
            "FSDP": fsdp, "FSDP_CPU_OFFLOAD": False, "XFuser_attention": attention,
            "skip_comm_test": False, "use_mmap": True, "ray_object_store_gb": 32.0 + ray_init_salt * 0.001}},
        "unet": {"class_type": "RayUNETLoader", "inputs": {
            "unet_name": unet, "weight_dtype": "default", "ray_actors_init": ["ray", 0]}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "minimax", "device": clip_device}},
        "vae_v": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "vae_a": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vae_v", 0], "prompt": prompt,
            "width": width, "height": height, "length": length}},
        "guider": {"class_type": "RayBasicGuider", "inputs": {"ray_actors": ["unet", 0], "conditioning": ["cond", 0]}},
        "sched": {"class_type": "RayBasicScheduler", "inputs": {
            "ray_actors": ["unet", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sample": {"class_type": "XFuserSamplerCustomAdvanced", "inputs": {
            "add_noise": True, "noise_seed": seed, "guider": ["guider", 0], "sampler": ["sampler", 0],
            "sigmas": ["sched", 0], "latent_image": ["cond", 1]}},
        "dec_v": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae_v", 0]}},
        "dec_a": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["vae_a", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["dec_v", 0], "audio": ["dec_a", 0], "fps": 24}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["video", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
    }
    if unet.endswith(".gguf"):
        wf["ray"]["inputs"]["FSDP"] = False
        wf["unet"] = {"class_type": "RayGGUFLoader", "inputs": {
            "unet_name": unet, "dequant_dtype": "default", "patch_dtype": "default", "ray_actors_init": ["ray", 0]}}
    if clip.endswith(".gguf"):
        wf["clip"] = {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": clip, "type": "minimax"}}
    if sla:
        wf["sla"] = {"class_type": "RayMiniMaxH3SLA", "inputs": {
            "ray_actors": ["unet", 0], "enabled": True, "sparsity_ratio": sla, "block_size": "64",
            "min_seq_len": 8192, "dense_last_steps": 1, "protect_audio": True}}
        for node in ("guider", "sched"):
            wf[node]["inputs"]["ray_actors"] = ["sla", 0]
    if lora:
        wf["lora"] = {"class_type": "RayLoraLoader", "inputs": {"lora_name": lora_name, "strength_model": 1.0}}
        wf["unet"]["inputs"]["lora"] = ["lora", 0]
    if guide_video:
        # Continuation: anchor the previous segment's tail (frames + soundtrack) at frame 0.
        wf["gload"] = {"class_type": "LoadVideo", "inputs": {"file": guide_video}}
        wf["gparts"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["gload", 0]}}
        g = {"positive": ["cond", 0], "latent": ["cond", 1], "frame_idx": 0,
             "vae": ["vae_v", 0], "image": ["gparts", 0]}
        if guide_frames_audio:
            g.update({"audio_vae": ["vae_a", 0], "audio": ["gparts", 1]})
        wf["guide"] = {"class_type": "MiniMaxH3AddGuide", "inputs": g}
        wf["guider"]["inputs"]["conditioning"] = ["guide", 0]
    return wf


def kill_ray(host="http://localhost:8190", ray_init_salt=0, **build_kwargs):
    """Shut down the Ray cluster so its workers release GPU memory.

    Uses the same RayInitializer inputs as the last job so ComfyUI's cache hands back the live actors.
    The next job must use a different ray_init_salt to force a fresh init."""
    full = build("", ray_init_salt=ray_init_salt, **build_kwargs)
    wf = {k: full[k] for k in ("ray", "unet", "lora") if k in full}  # cached -> no reload
    wf["kill"] = {"class_type": "RayKill", "inputs": {"ray_actors": ["unet", 0], "kill_mode": "Kill Entire Cluster"}}
    run(wf, host)


def run(wf, host="http://localhost:8190", poll=5):
    """Submit, wait, return list of output file paths (relative to the ComfyUI root). Raises on error."""
    def call(path, body=None):
        req = urllib.request.Request(host + path, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req))
    try:
        pid = call("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode())
    t0 = time.time()
    while True:
        h = call(f"/history/{pid}").get(pid)
        if h and h.get("status", {}).get("completed") is not None:
            break
        time.sleep(poll)
    st = h["status"]
    if st["status_str"] != "success":
        errs = [m[1] for m in st.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
        raise RuntimeError(json.dumps(errs, indent=1)[:4000])
    outs = []
    for node in h.get("outputs", {}).values():
        for v in node.values():
            for item in v if isinstance(v, list) else []:
                # Skip LoadVideo/preview entries (type "input"/"temp"); keep files actually saved to output/.
                if isinstance(item, dict) and "filename" in item and item.get("type", "output") == "output":
                    outs.append("/".join(x for x in ("output", item.get("subfolder", ""), item["filename"]) if x))
    print(f"  done in {time.time() - t0:.0f}s -> {outs}")
    return outs
