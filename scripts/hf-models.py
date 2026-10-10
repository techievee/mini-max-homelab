#!/usr/bin/env python3
"""MiniMax H3 model files: keep them in the Hugging Face cache, expose them to ComfyUI through symlinks.

ComfyUI wants   mini-max-amd-7900xtx/models/<folder>/<file>          (diffusion_models, text_encoders, vae, loras)
the HF cache has ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<commit>/<path>  ->  ../../blobs/<sha256>
This tool makes every models/<folder>/<file> an absolute symlink into the cache snapshot. The compose files mount the
HF hub cache into the containers at the *same* absolute path, so the links resolve inside Docker too.

Commands (run on the host, needs `pip install huggingface_hub`):
  status              what is linked, cached, or still a plain file
  download [NAME..]   download into the HF cache (skips what is already cached), then link     (default: the "default" set)
  link     [NAME..]   (re)create symlinks for files already in the cache
  adopt    [NAME..]   one-time migration: verify SHA-256 of existing plain files in models/ against the Hub, move them into
                      the cache (no re-download, same filesystem = instant move), then link
NAME = a key from `status`, a set name (default, all), or omitted for "default".

Env: MODELS_DIR (default: <repo>/mini-max-amd-7900xtx/models), HF_HUB_CACHE / HF_HOME (standard huggingface_hub variables), HF_TOKEN.
"""
import argparse
import hashlib
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", REPO_ROOT / "mini-max-amd-7900xtx" / "models"))

# key: (hf repo, path in repo, ComfyUI folder, sets)
MANIFEST = {
    "dit-bf16":    ("Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors", "diffusion_models", {"default"}),
    "te-bf16":     ("Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors", "text_encoders", {"default"}),
    "vae-video":   ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", "vae", {"default"}),
    "vae-audio":   ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors", "vae", {"default"}),
    "lora-4step":  ("Comfy-Org/MiniMax-H3", "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", "loras", {"default"}),
    "lora-8step":  ("Comfy-Org/MiniMax-H3", "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "loras", {"default"}),
    "dit-q4":      ("unsloth/MiniMax-H3-GGUF", "minimax_h3_fl2va_pruned-Q4_K.gguf", "diffusion_models", {"default", "q4"}),
    "te-q4":       ("unsloth/MiniMax-H3-GGUF", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "text_encoders", {"default", "q4"}),
    # optional extras (not in "default"):
    "dit-fp8":     ("Comfy-Org/MiniMax-H3", "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "diffusion_models", {"small"}),
    "te-nvfp4":    ("Comfy-Org/MiniMax-H3", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders", {"small"}),
}


def hub():
    try:
        import huggingface_hub
        from huggingface_hub import constants
    except ImportError:
        sys.exit("needs huggingface_hub:  pip install -U huggingface_hub")
    return huggingface_hub, Path(constants.HF_HUB_CACHE)


def select(names):
    names = names or ["default"]
    keys = []
    for n in names:
        if n in MANIFEST:
            keys.append(n)
        elif n == "all":
            keys += list(MANIFEST)
        elif any(n in sets for *_, sets in MANIFEST.values()):
            keys += [k for k, (*_, sets) in MANIFEST.items() if n in sets]
        else:
            sys.exit(f"unknown model or set: {n}  (see `status`)")
    return list(dict.fromkeys(keys))


def link_path(key):
    repo, path, folder, _ = MANIFEST[key]
    return MODELS_DIR / folder / Path(path).name


def cached_path(key):
    """Snapshot path of the file in the HF cache, or None (no network)."""
    hf, _ = hub()
    repo, path, *_ = MANIFEST[key]
    try:
        return Path(hf.hf_hub_download(repo, path, local_files_only=True))
    except Exception:
        return None


def make_link(key):
    target = cached_path(key)
    dst = link_path(key)
    if target is None:
        print(f"  {key:11} not in HF cache - run `download {key}`")
        return False
    if dst.exists() and not dst.is_symlink():
        print(f"  {key:11} {dst} is a plain file - run `adopt {key}` to move it into the cache")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    dst.symlink_to(target)  # absolute: the compose files mount the HF cache at the same path in the containers
    print(f"  {key:11} {dst.relative_to(MODELS_DIR)} -> {target}")
    return True


def cmd_status(_):
    _, cache = hub()
    print(f"models dir: {MODELS_DIR}\nHF cache:   {cache}\n")
    for key, (repo, path, folder, sets) in MANIFEST.items():
        dst = link_path(key)
        if dst.is_symlink():
            state = "linked" if dst.exists() else "BROKEN LINK"
        elif dst.exists():
            state = "plain file (adopt)"
        else:
            state = "-"
        cached = "cached" if cached_path(key) else "-"
        print(f"  {key:11} {','.join(sorted(sets)):13} {state:19} {cached:7} {repo}/{path}")


def cmd_download(a):
    hf, _ = hub()
    for key in select(a.names):
        repo, path, *_ = MANIFEST[key]
        if cached_path(key) is None:
            print(f"downloading {repo}/{path}")
            hf.hf_hub_download(repo, path)
        make_link(key)


def cmd_link(a):
    for key in select(a.names):
        make_link(key)


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def cmd_adopt(a):
    hf, cache = hub()
    api = hf.HfApi()
    todo = [k for k in select(a.names) if link_path(k).exists() and not link_path(k).is_symlink()]
    if not todo:
        print("nothing to adopt (no plain files in models/ for the selected keys)")
        return
    meta = {}
    for repo in {MANIFEST[k][0] for k in todo}:
        info = api.model_info(repo, files_metadata=True)
        meta[repo] = (info.sha, {s.rfilename: s for s in info.siblings})
    print(f"hashing {len(todo)} file(s) in parallel ...")
    with ProcessPoolExecutor(max_workers=len(todo)) as ex:
        digests = dict(zip(todo, ex.map(_sha256, [link_path(k) for k in todo])))
    for key in todo:
        repo, path, *_ = MANIFEST[key]
        commit, siblings = meta[repo]
        sib = siblings.get(path)
        src = link_path(key)
        if sib is None or sib.lfs is None:
            print(f"  {key:11} not found on the Hub as an LFS file - skipped"); continue
        if digests[key] != sib.lfs.sha256 or src.stat().st_size != sib.size:
            print(f"  {key:11} SHA-256/size mismatch with {repo}@{commit[:8]} - skipped (file left in place)"); continue
        repo_dir = cache / ("models--" + repo.replace("/", "--"))
        blob = repo_dir / "blobs" / sib.lfs.sha256
        snap = repo_dir / "snapshots" / commit / path
        blob.parent.mkdir(parents=True, exist_ok=True)
        snap.parent.mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs").mkdir(exist_ok=True)
        (repo_dir / "refs" / "main").write_text(commit)
        if blob.exists():
            src.unlink()  # identical content already cached
        else:
            shutil.move(src, blob)  # same filesystem -> rename
        if snap.is_symlink() or snap.exists():
            snap.unlink()
        snap.symlink_to(os.path.relpath(blob, snap.parent))  # the HF cache's own relative layout
        print(f"  {key:11} verified + moved into cache ({sib.size / 1e9:.1f} GB)")
        make_link(key)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("status", cmd_status), ("download", cmd_download), ("link", cmd_link), ("adopt", cmd_adopt)]:
        sp = sub.add_parser(name)
        if name != "status":
            sp.add_argument("names", nargs="*")
        sp.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
