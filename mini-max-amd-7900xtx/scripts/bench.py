"""Benchmark MiniMax H3 configurations on the 4x RX 7900 XTX (bench container, port 8192).

Runs each config through ComfyUI, samples GPU busy % / VRAM from sysfs (host) every second, and splits the run into
phases: text encoder + load (before all 4 GPUs are busy), sampling (all 4 busy), decode (after). Appends one JSON line
per run to output/bench/results.jsonl.

Usage (host):  python3 scripts/bench.py screen|full [--only NAME ...] [--host http://localhost:8192]
"""
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_workflow as h3

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# drm card numbers shift after BIOS/hardware changes (card3-6 on 2026-10-10, card2-5 after the 2026-10-10 BIOS reset),
# so resolve them from the PCI address. Order = cuda:0..3.
CARDS = [os.path.basename(os.path.realpath(f"/dev/dri/by-path/pci-0000:{b}:00.0-card")) for b in ("48", "4e", "53", "56")]

# Same prompt + seed as the NVIDIA runs (../mini-max-nvidia-3090, h3_t2v_raylight_cuda8*.json).
_ui = os.path.join(ROOT, "..", "mini-max-nvidia-3090", "user", "default", "api_workflows", "api_h3_t2v_raylight_cuda8_hq.json")
_ref = json.load(open(_ui))
PROMPT, SEED = _ref["cond"]["inputs"]["prompt"], _ref["sample"]["inputs"]["noise_seed"]

BF16 = dict()
Q4 = dict(unet=h3.Q4_UNET, clip=h3.Q4_CLIP)
Q4TE = dict(clip=h3.Q4_CLIP)
SHORT = dict(width=1344, height=768, length=124)   # 5.2 s screening clip
FULL = dict(width=1344, height=768, length=362)    # 15.1 s, same as the NVIDIA runs

CONFIGS = {
    "screen": [
        ("s1_bf16_fsdp_u4_flash_bf16te", dict(**BF16, **SHORT, steps=4)),
        ("s2_q4_u4_flash_q4te", dict(**Q4, **SHORT, steps=4)),
        ("s3_bf16_fsdp_u4_q4te", dict(**Q4TE, **SHORT, steps=4)),
        ("s4_bf16_fsdp_u2r2_q4te", dict(**Q4TE, **SHORT, steps=4, ring=2)),
        ("s5_q4_u4_sla075", dict(**Q4, **SHORT, steps=4, sla=0.75)),
        ("s6_q4_u4_sla090", dict(**Q4, **SHORT, steps=4, sla=0.90)),
    ],
    "full": [],  # filled from the screening winner via --best
}


def sample_gpus(stop, rows):
    def rd(card, f):
        try:
            return int(open(f"/sys/class/drm/{card}/device/{f}").read())
        except OSError:
            return -1
    while not stop.is_set():
        mem = open("/proc/meminfo").read()
        tot = int(re.search(r"MemTotal:\s+(\d+)", mem).group(1)); av = int(re.search(r"MemAvailable:\s+(\d+)", mem).group(1))
        sw = int(re.search(r"SwapTotal:\s+(\d+)", mem).group(1)) - int(re.search(r"SwapFree:\s+(\d+)", mem).group(1))
        rows.append((time.time(), [rd(c, "gpu_busy_percent") for c in CARDS],
                     [rd(c, "mem_info_vram_used") // 2**20 for c in CARDS], (tot - av) / 2**20, sw / 2**10))
        time.sleep(1)


def phases(rows, t0, t1):
    busy = [r for r in rows if min(r[1]) >= 80]
    if not busy:
        return dict(pre=None, sampling=None, post=None)
    s0, s1 = busy[0][0], busy[-1][0]
    return dict(pre=round(s0 - t0), sampling=round(s1 - s0), post=round(t1 - s1))


def check_output(path):
    """Mean luma over all frames + a mid/late frame for eyeballing. Black/NaN runs come out as a ~100 KB uniform clip."""
    full = os.path.join(ROOT, path)
    y = subprocess.run(["bash", "-c", f"ffmpeg -v error -i '{full}' -vf signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=- "
                        "-f null - | grep -o 'YAVG=[0-9.]*' | cut -d= -f2 | awk '{s+=$1;n++} END{printf \"%.1f\", s/n}'"],
                       capture_output=True, text=True).stdout
    mean_y = float(y or 0); size = os.path.getsize(full)
    frames = []
    for t in (2, 6, 12):
        out = os.path.join(ROOT, "output", "bench", "frames", f"{os.path.basename(full)[:-4]}_{t}s.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", full, "-frames:v", "1", "-vf", "scale=672:-1", out]).returncode == 0 \
                and os.path.exists(out):
            frames.append(out)
    nframes = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                  "stream=nb_read_frames", "-of", "csv=p=0", full], capture_output=True, text=True).stdout.strip() or 1)
    kb_per_frame = size / nframes / 1024
    # Black/NaN clips: ~0.3 KB/frame, mean luma 16. Noise/garbage clips (seen after mixing FSDP and GGUF runs): >300 KB/frame
    # (incompressible). Good 1344x768 / 864x480 clips measured 23-125 KB/frame... 125 was the noise one at 480p, so cap at 100.
    # Per-frame size instead of an absolute minimum, so short smoke-test clips aren't flagged.
    good = mean_y >= 25 and 3 <= kb_per_frame <= 100
    return dict(good=good, mean_y=mean_y, size=size, kb_per_frame=round(kb_per_frame, 1), frames=frames)


def restart_container():
    print("  -> restarting mini-max-amd-7900xtx-bench (clears stale FSDP/GGUF state)", flush=True)
    subprocess.run(["docker", "compose", "--profile", "bench", "restart", "comfyui-bench"], cwd=ROOT, check=True)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://localhost:8192/system_stats", timeout=3); return
        except Exception:
            time.sleep(3)


def run_one(name, kw, host, salt):
    wf = h3.build(PROMPT, seed=SEED, prefix=f"video/bench/{name}", ray_init_salt=salt, **kw)
    rows, stop = [], threading.Event()
    th = threading.Thread(target=sample_gpus, args=(stop, rows), daemon=True); th.start()
    t0 = time.time(); err = None; outs = []
    try:
        outs = h3.run(wf, host, poll=5)
    except Exception as e:  # keep benchmarking the rest
        err = str(e)[:2000]
    t1 = time.time(); stop.set(); th.join()
    ph = phases(rows, t0, t1)
    chk = check_output(next(o for o in outs if o.endswith(".mp4"))) if outs else dict(good=False, mean_y=0, size=0, frames=[])
    res = dict(name=name, config={k: v for k, v in kw.items()}, ok=err is None and chk["good"], error=err, outputs=outs,
               output_check=chk,
               total_s=round(t1 - t0), **{f"{k}_s": v for k, v in ph.items()},
               per_step_s=round(ph["sampling"] / kw["steps"], 1) if ph["sampling"] else None,
               peak_vram_mib=[max(r[2][i] for r in rows) for i in range(4)] if rows else None,
               peak_host_gib=round(max(r[3] for r in rows), 1) if rows else None,
               peak_swap_mib=round(max(r[4] for r in rows)) if rows else None,
               when=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)))
    os.makedirs(os.path.join(ROOT, "output", "bench"), exist_ok=True)
    with open(os.path.join(ROOT, "output", "bench", "results.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: res[k] for k in ("name", "ok", "total_s", "pre_s", "sampling_s", "post_s", "per_step_s",
                                          "peak_vram_mib", "peak_host_gib", "peak_swap_mib")}), flush=True)
    print(f"  output check: {'GOOD' if chk['good'] else 'BAD'} meanY={chk['mean_y']} size={chk['size']} frames={chk['frames']}", flush=True)
    if err:
        print("  ERROR:", err[:600], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=list(CONFIGS) + ["custom"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--host", default="http://localhost:8192")
    ap.add_argument("--custom", help='JSON list of [name, kwargs] pairs for suite "custom"')
    a = ap.parse_args()
    cfgs = json.loads(a.custom) if a.suite == "custom" else CONFIGS[a.suite]
    last_family = None
    for i, (name, kw) in enumerate(cfgs):
        if a.only and name not in a.only:
            continue
        # Mixing bf16+FSDP and GGUF runs in one container corrupts the next run (black or noise). Restart on every switch.
        family = "gguf" if str(kw.get("unet", "")).endswith(".gguf") else "bf16"
        if last_family is not None and family != last_family:
            restart_container()
        last_family = family
        print(f"=== {name}: {kw}", flush=True)
        res = run_one(name, kw, a.host, salt=int(time.time()) % 100000 + i)
        if not res["ok"]:
            restart_container()
            print(f"=== {name} (retry after restart)", flush=True)
            res = run_one(name + "_retry", kw, a.host, salt=int(time.time()) % 100000 + i + 50)
            if not res["ok"]:
                sys.exit(f"STOP: {name} still bad after a container restart -- not starting the next test")
