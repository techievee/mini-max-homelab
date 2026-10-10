"""Make a long MiniMax H3 clip (e.g. 30 s) by chaining segments inside the model's trained range (~5-15 s).

Segment k>1 is anchored (MiniMaxH3AddGuide, frame 0) on the last OVERLAP frames + audio of segment k-1,
then the overlap is dropped and everything is concatenated into one mp4.

Run inside the container (its ffmpeg has libx264; Fedora's does not):
  docker exec mini-max-nvidia-3090 python scripts/run-long.py [--seconds 30] [--segment-seconds 10.8] [--story story.json]
  --existing output/video/x_seg1_00001_.mp4 ...   reuse already-generated segments (resume after a failure)
story.json: {"style": "...", "beats": ["segment 1 action ...", "segment 2 action ...", ...]}
"""
import argparse, json, math, os, subprocess
import h3_workflow as h3

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OVERLAP = 22  # frames carried into the next segment (a valid 17k+5 clip length)

DEFAULT_STORY = {
    "style": ("Cinematic live-action road movie, anamorphic lens, warm golden-hour light, film grain, "
              "consistent character: a rider in a black helmet, black leather jacket and blue jeans "
              "on a red vintage motorcycle. Natural stereo sound design."),
    "beats": [
        "The rider cruises along a wet coastal road at sunset, waves crashing against black rocks beside the road. "
        "Tracking shot alongside the bike. Sound: engine rumble, waves, tires on wet asphalt.",
        "The rider continues into a small seaside village, passing white-washed houses, string lights and a cafe "
        "with people waving. Camera follows from behind, then pans alongside. Sound: engine, distant chatter, "
        "seagulls, light music from the cafe.",
        "The rider climbs a winding cliff road and stops at an overlook above the ocean, kills the engine, takes off "
        "the helmet and watches the sun touch the horizon. Slow push-in. Sound: engine winding down, wind, "
        "distant surf, calm silence.",
    ],
}


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def nframes(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                         check=True, capture_output=True, text=True).stdout
    return int(out.strip())


p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=30)
p.add_argument("--segment-seconds", type=float, default=10.8)
p.add_argument("--story", help="JSON file with style + beats (one beat per segment, repeated if short)")
p.add_argument("--width", type=int, default=864)
p.add_argument("--height", type=int, default=480)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--name", default="long")
p.add_argument("--clip-device", default="cpu", choices=["default", "cpu"],
               help="cpu keeps GPU0 free for VAE-encoding the continuation frames (avoids OOM next to the Ray worker)")
p.add_argument("--existing", nargs="*", default=[], help="already-generated segment mp4s (relative to ComfyUI root)")
a = p.parse_args()

story = json.load(open(a.story)) if a.story else DEFAULT_STORY
seg_len = h3.snap_length(a.segment_seconds)
new_per_seg = seg_len - OVERLAP
n_seg = max(1, 1 + math.ceil((a.seconds * 24 - seg_len) / new_per_seg))
total = seg_len + (n_seg - 1) * new_per_seg
print(f"{n_seg} segments x {seg_len} frames ({seg_len/24:.2f}s), overlap {OVERLAP} -> "
      f"{total} frames ({total/24:.2f}s) at {a.width}x{a.height}")

segments = list(a.existing)
for k in range(len(segments), n_seg):
    beat = story["beats"][min(k, len(story["beats"]) - 1)]
    prompt = f"{story['style']}\n\n{beat}"
    guide = None
    if k:
        # Last OVERLAP frames + matching audio of the previous segment -> input/ for LoadVideo.
        prev = os.path.join(ROOT, segments[-1])
        start = nframes(prev) - OVERLAP
        guide = f"{a.name}_tail_{k}.mp4"
        ff("-i", prev, "-vf", f"trim=start_frame={start},setpts=PTS-STARTPTS",
           "-af", f"atrim=start={start/24},asetpts=PTS-STARTPTS",
           "-c:v", "libx264", "-crf", "12", "-c:a", "aac", os.path.join(ROOT, "input", guide))
    print(f"segment {k+1}/{n_seg}{' (continuing from ' + guide + ')' if guide else ''}")
    wf = h3.build(prompt, a.width, a.height, seg_len, a.steps, a.seed + k,
                  guide_video=guide, clip_device=a.clip_device, prefix=f"video/{a.name}_seg{k+1}")
    segments.append([f for f in h3.run(wf) if f.endswith(".mp4")][0])

# Stitch: first segment whole, later segments without their first OVERLAP frames (they duplicate the tail).
inputs, chains, labels = [], [], ""
for i, s in enumerate(segments):
    inputs += ["-i", os.path.join(ROOT, s)]
    sf = 0 if i == 0 else OVERLAP
    chains.append(f"[{i}:v]trim=start_frame={sf},setpts=PTS-STARTPTS[v{i}];"
                  f"[{i}:a]atrim=start={sf/24},asetpts=PTS-STARTPTS[a{i}]")
    labels += f"[v{i}][a{i}]"
final = os.path.join(ROOT, "output", "video", f"{a.name}_{total/24:.0f}s.mp4")
ff(*inputs, "-filter_complex", ";".join(chains) + f";{labels}concat=n={len(segments)}:v=1:a=1[v][a]",
   "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-c:a", "aac", "-b:a", "192k",
   final)
print("final:", os.path.relpath(final, ROOT))
