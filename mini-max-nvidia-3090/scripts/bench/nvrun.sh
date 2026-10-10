#!/usr/bin/env bash
# 4x3090 PLX video benchmarks after the iommu=pt / ACS change. Restart before each run, check output after; stop if bad.
S=${BENCH_TMP:-/tmp/mini-max-bench}; mkdir -p "$S"   # scratch dir for logs
P=$(cd "$(dirname "$0")/../.." && pwd)                  # project root (mini-max-nvidia-3090)
for key in "$@"; do
  prefix=$(python3 -c "import json; print(json.load(open('$P/user/default/api_workflows/bench_$key.json'))['save']['inputs']['filename_prefix'])")
  before=$(ls -t $P/output/$prefix* 2>/dev/null | head -1)
  (cd $P && docker compose --profile cuda8 restart comfyui-cuda8 >/dev/null 2>&1); until curl -fs localhost:8191/system_stats >/dev/null; do sleep 3; done
  echo "=== $key ($prefix)"
  $S/pwtest.sh "$key" acsoff
  f=$(ls -t $P/output/$prefix* 2>/dev/null | head -1)
  if [ -z "$f" ] || [ "$f" = "$before" ]; then echo "  BAD: no new output"; exit 1; fi
  n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$f")
  y=$(ffmpeg -v error -i "$f" -vf signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=- -f null - | grep -o 'YAVG=[0-9.]*' | cut -d= -f2 | awk '{s+=$1;c++} END{printf "%.1f", s/c}')
  kb=$(( $(stat -c%s "$f") / n / 1024 ))
  mkdir -p $S/cmp; ffmpeg -v error -y -ss 6 -i "$f" -frames:v 1 -vf scale=672:-1 $S/cmp/nv_acsoff_$key.png
  if awk "BEGIN{exit !($y >= 25 && $kb <= 100 && $kb >= 3)}"; then echo "  output GOOD: $(basename $f) frames=$n meanY=$y KB/frame=$kb"
  else echo "  output BAD: $(basename $f) frames=$n meanY=$y KB/frame=$kb -> stopping"; exit 1; fi
done
