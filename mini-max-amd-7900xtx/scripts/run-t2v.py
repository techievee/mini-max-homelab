"""Submit a MiniMax H3 text-to-video job to ComfyUI, split across GPUs with Raylight.

Usage (from the host):
  python3 scripts/run-t2v.py --prompt "..." [--seconds 2] [--steps 4] [--gpus 4] [--no-lora]
Writes the API workflow it submitted to user/default/workflows/api_h3_t2v_raylight.json.
"""
import argparse, json, os
import h3_workflow as h3

p = argparse.ArgumentParser()
p.add_argument("--host", default="http://localhost:8190")
p.add_argument("--prompt", default=(
    "Cinematic live-action shot: a red vintage motorcycle rides along a wet coastal road at sunset, "
    "waves crashing against black rocks, warm golden backlight, light rain, shallow depth of field. "
    "Sound: engine rumble, waves, light rain on asphalt."))
p.add_argument("--seconds", type=float, default=2.0)
p.add_argument("--width", type=int, default=864)
p.add_argument("--height", type=int, default=480)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--gpus", type=int, default=4)
p.add_argument("--ulysses", type=int, default=None, help="defaults to --gpus")
p.add_argument("--no-fsdp", action="store_true")
p.add_argument("--no-lora", action="store_true", help="base model, 20 steps recommended")
p.add_argument("--clip-device", default="default", choices=["default", "cpu"])
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

length = h3.snap_length(a.seconds)
wf = h3.build(a.prompt, a.width, a.height, length, a.steps, a.seed, a.gpus, a.ulysses,
              fsdp=not a.no_fsdp, lora=not a.no_lora, clip_device=a.clip_device)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "user", "default", "workflows")
os.makedirs(out, exist_ok=True)
json.dump(wf, open(os.path.join(out, "api_h3_t2v_raylight.json"), "w"), indent=2)

print(f"{a.width}x{a.height}, {length} frames ({length/24:.2f}s), {a.steps} steps, "
      f"gpus={a.gpus} ulysses={a.ulysses or a.gpus} fsdp={not a.no_fsdp} lora={not a.no_lora}")
for f in h3.run(wf, a.host):
    print("output:", f)
