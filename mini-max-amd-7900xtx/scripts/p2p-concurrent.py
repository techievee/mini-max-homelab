# Concurrent P2P copies: if aggregate stays ~one-link speed, peer traffic is funnelled through a shared
# upstream link (ACS redirect to the root port) instead of being switched locally.
import torch, time
size = 512 * 2**20
def run(pairs, iters=10):
    bufs = [(torch.empty(size, dtype=torch.uint8, device=s), torch.empty(size, dtype=torch.uint8, device=d)) for s, d in pairs]
    streams = [torch.cuda.Stream(device=s) for s, _ in pairs]
    for (a, b), st in zip(bufs, streams):
        with torch.cuda.stream(st): b.copy_(a, non_blocking=True)
    for i in range(4): torch.cuda.synchronize(i)
    t = time.perf_counter()
    for _ in range(iters):
        for (a, b), st in zip(bufs, streams):
            with torch.cuda.stream(st): b.copy_(a, non_blocking=True)
    for i in range(4): torch.cuda.synchronize(i)
    return size * iters * len(pairs) / (time.perf_counter() - t) / 1e9
for name, pairs in [("0->1", [(0, 1)]), ("0->1 + 2->3", [(0, 1), (2, 3)]),
                    ("0->1 1->0 (bidir)", [(0, 1), (1, 0)]),
                    ("ring 0->1->2->3->0", [(0, 1), (1, 2), (2, 3), (3, 0)])]:
    print(f"{name:22s} aggregate {run(pairs):6.1f} GB/s")
