# Sanity check for the 4-GPU setup: device visibility, P2P access, RCCL all-reduce bandwidth.
# Run: docker compose run --rm --entrypoint torchrun comfyui --nproc_per_node 4 scripts/gpu-check.py
import os, time, torch, torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")

if rank == 0:
    print(f"torch {torch.__version__} hip {torch.version.hip}, {torch.cuda.device_count()} devices")
    for i in range(torch.cuda.device_count()):
        peers = [j for j in range(torch.cuda.device_count()) if j != i and torch.cuda.can_device_access_peer(i, j)]
        print(f"  [{i}] {torch.cuda.get_device_name(i)}  p2p-> {peers}")

x = torch.ones(256 * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")  # 256 MiB
for _ in range(3):
    dist.all_reduce(x)
torch.cuda.synchronize()
iters = int(os.environ.get("ITERS", 10))
t = time.perf_counter()
for _ in range(iters):
    dist.all_reduce(x)
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / iters
bus_bw = x.numel() * x.element_size() * 2 * (world - 1) / world / dt / 1e9
if rank == 0:
    print(f"all_reduce 256 MiB x{world}: {dt*1e3:.1f} ms, bus bw {bus_bw:.1f} GB/s")
dist.destroy_process_group()
