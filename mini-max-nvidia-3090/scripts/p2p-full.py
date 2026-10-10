# Full peer-to-peer test for all visible GPUs: per-pair uni/bidirectional bandwidth (two sizes), latency, and
# concurrent patterns (disjoint pairs, ring, all-to-all). Synchronizes every involved device before timing.
# Run: docker compose run --rm --entrypoint python comfyui scripts/p2p-full.py
import itertools, time, torch

n = torch.cuda.device_count()
bus = [f"{torch.cuda.get_device_properties(i).pci_bus_id:02x}" for i in range(n)]
print(f"{n} GPUs: " + ", ".join(f"{i}={b}:00.0" for i, b in enumerate(bus)))
print("peer access:", {i: [j for j in range(n) if j != i and torch.cuda.can_device_access_peer(i, j)] for i in range(n)})


def sync(devs):
    for d in devs:
        torch.cuda.synchronize(d)


def timed(fn, devs, nbytes, iters):
    fn(); sync(devs)
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(devs)
    return nbytes * iters / (time.perf_counter() - t) / 1e9


def bufs(size):
    return {i: (torch.empty(size, dtype=torch.uint8, device=i), torch.empty(size, dtype=torch.uint8, device=i)) for i in range(n)}


def copies(pairs, b):
    """Launch one copy per (src, dst) on a stream of the source device."""
    st = {p: torch.cuda.Stream(device=p[0]) for p in pairs}

    def go():
        for s, d in pairs:
            with torch.cuda.stream(st[(s, d)]):
                b[d][1].copy_(b[s][0], non_blocking=True)
    return go


for size, iters in ((16 * 2**20, 50), (512 * 2**20, 10)):
    b = bufs(size)
    label = f"{size // 2**20} MiB"
    print(f"\n== unidirectional GB/s ({label}), row = src, col = dst")
    print("      " + "".join(f"{bus[j]:>8}" for j in range(n)))
    for i in range(n):
        print(f"{bus[i]:>4}  " + "".join("       -" if i == j else f"{timed(copies([(i, j)], b), [i, j], size, iters):8.1f}" for j in range(n)))
    print(f"== bidirectional GB/s ({label}), both directions at once, total")
    for i, j in itertools.combinations(range(n), 2):
        print(f"  {bus[i]}<->{bus[j]}: {timed(copies([(i, j), (j, i)], b), [i, j], 2 * size, iters):6.1f}")
    del b
    torch.cuda.empty_cache()

# Latency: 4 KiB copies, mean of 200
b = bufs(4096)
print("\n== latency (4 KiB copy, µs), row = src, col = dst")
print("      " + "".join(f"{bus[j]:>8}" for j in range(n)))
for i in range(n):
    row = []
    for j in range(n):
        if i == j:
            row.append("       -"); continue
        f = copies([(i, j)], b); f(); sync([i, j])
        t = time.perf_counter()
        for _ in range(200):
            f(); sync([i, j])
        row.append(f"{(time.perf_counter() - t) / 200 * 1e6:8.1f}")
    print(f"{bus[i]:>4}  " + "".join(row))

size, iters = 512 * 2**20, 10
b = bufs(size)
patterns = {
    "two disjoint pairs 0->1 + 2->3": [(0, 1), (2, 3)],
    "two disjoint pairs 0->2 + 1->3": [(0, 2), (1, 3)],
    "bidir pairs 0<->1 + 2<->3": [(0, 1), (1, 0), (2, 3), (3, 2)],
    "ring 0->1->2->3->0": [(i, (i + 1) % n) for i in range(n)],
    "all-to-all (every ordered pair)": [(i, j) for i in range(n) for j in range(n) if i != j],
    "one-to-all 0->1,2,3": [(0, j) for j in range(1, n)],
    "all-to-one 1,2,3->0": [(j, 0) for j in range(1, n)],
}
print(f"\n== concurrent patterns ({size // 2**20} MiB per copy), aggregate GB/s")
for name, pairs in patterns.items():
    pairs = [p for p in pairs if max(p) < n]
    print(f"  {name:34s} {timed(copies(pairs, b), range(n), size * len(pairs), iters):6.1f}")
