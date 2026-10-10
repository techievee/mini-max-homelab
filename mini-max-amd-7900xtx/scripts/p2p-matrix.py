# Unidirectional device-to-device copy bandwidth (GB/s) for every GPU pair, plus host<->device.
import torch, time
n = torch.cuda.device_count()
size = 512 * 2**20
def bw(src, dst):
    a = torch.empty(size, dtype=torch.uint8, device=src); b = torch.empty(size, dtype=torch.uint8, device=dst)
    for _ in range(2): b.copy_(a)
    torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    t = time.perf_counter()
    for _ in range(10): b.copy_(a)
    torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    return size * 10 / (time.perf_counter() - t) / 1e9
print("src\\dst " + " ".join(f"{j:>6}" for j in range(n)) + "   H2D    D2H")
for i in range(n):
    row = [("     -" if i == j else f"{bw(f'cuda:{i}', f'cuda:{j}'):6.1f}") for j in range(n)]
    h2d = bw("cpu", f"cuda:{i}") if False else None
    pin = torch.empty(size, dtype=torch.uint8).pin_memory(); d = torch.empty(size, dtype=torch.uint8, device=i)
    torch.cuda.synchronize(i); t = time.perf_counter()
    for _ in range(10): d.copy_(pin, non_blocking=True)
    torch.cuda.synchronize(i); h2d = size * 10 / (time.perf_counter() - t) / 1e9
    t = time.perf_counter()
    for _ in range(10): pin.copy_(d, non_blocking=True)
    torch.cuda.synchronize(i); d2h = size * 10 / (time.perf_counter() - t) / 1e9
    print(f"{i:>7} " + " ".join(row) + f" {h2d:6.1f} {d2h:6.1f}")
