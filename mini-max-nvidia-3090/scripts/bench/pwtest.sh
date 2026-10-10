#!/usr/bin/env bash
# Run one NVIDIA bench workflow on :8191 with per-second GPU util/power/clock sampling.
# $1 = workflow key (api_workflows/bench_<key>.json), $2 = label. Prints total, phases, sampling, power.
S=${BENCH_TMP:-/tmp/mini-max-bench}; mkdir -p "$S"   # scratch dir for logs
P=$(cd "$(dirname "$0")/../.." && pwd)                  # project root (mini-max-nvidia-3090)
L=$S/pw_$2_$1.log; : > $L
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used --format=csv,noheader,nounits | tr '\n' '|')" >> $L; sleep 1; done ) & P=$!
cd "$P" && T0=$(date +%s) && python3 -c "
import sys, json; sys.path.insert(0, 'scripts'); import h3_workflow as h3
h3.run(json.load(open('user/default/api_workflows/bench_$1.json')), host='http://localhost:8191', poll=2)" >/dev/null 2>&1; RC=$?; T1=$(date +%s); kill $P
python3 - "$L" "$1" "$2" $T0 $T1 $RC <<'PY'
import sys, json
L, key, label, t0, t1, rc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
steps = json.load(open(f"user/default/api_workflows/bench_{key}.json"))["sched"]["inputs"]["steps"]
need = 8 if key in ("pw_q2", "n1") else 4
rows = []
for l in open(L):
    ts, rest = l.split(" ", 1)
    g = [x.split(", ") for x in rest.strip().strip("|").split("|")]
    rows.append((int(ts), [int(a) for a, *_ in g], [float(b) for _, b, *_ in g], [int(c) for _, _, c, _ in g], [int(d) for *_, d in g]))
busy = [r for r in rows if sum(u >= 80 for u in r[1]) >= need]
if not busy:
    print(f"{key} {label} rc={rc} total={t1-t0}s  (no sampling phase detected)"); sys.exit()
b0, b1 = busy[0][0], busy[-1][0]
act = lambda r: [i for i, u in enumerate(r[1]) if u >= 80]
pw = [sum(r[2][i] for i in act(r)) / len(act(r)) for r in busy]
clk = [sum(r[3][i] for i in act(r)) / len(act(r)) for r in busy]
print(f"{key} {label} rc={rc} total={t1-t0}s pre={b0-t0}s sampling={b1-b0}s per_step={(b1-b0)/steps:.1f}s post={t1-b1}s "
      f"avg W/GPU={sum(pw)/len(pw):.0f} avg MHz={sum(clk)/len(clk):.0f} peak MiB={max(max(r[4]) for r in rows)}")
PY
