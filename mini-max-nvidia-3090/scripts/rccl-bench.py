# Collective bandwidth (the ops Raylight uses: FSDP all-gather / reduce-scatter, Ulysses all-to-all) at several sizes.
# Run: docker compose run --rm --entrypoint torchrun comfyui --nproc_per_node 4 scripts/rccl-bench.py   (NCCL on NVIDIA, RCCL on ROCm)
# Bus bandwidth follows nccl-tests conventions so numbers compare across GPU counts. BENCH_MIBS=256 to pick sizes.
import os, time, torch, torch.distributed as dist

rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
dev = torch.device("cuda", rank)


def bench(name, fn, nbytes, factor, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); dist.barrier()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    if rank == 0:
        print(f"  {name:15s} {nbytes / 2**20:7.0f} MiB  {dt * 1e3:8.2f} ms  algbw {nbytes / dt / 1e9:6.1f}  busbw {nbytes / dt / 1e9 * factor:6.1f} GB/s", flush=True)


if rank == 0:
    print(f"world={world}, backend={dist.get_backend()}, torch {torch.__version__}")
for mib in [int(x) for x in os.environ.get("BENCH_MIBS", "16,64,256").split(",")]:
    nbytes = mib * 2**20; elems = nbytes // 2
    x = torch.ones(elems, dtype=torch.bfloat16, device=dev)
    shard = torch.ones(elems // world, dtype=torch.bfloat16, device=dev)
    out = torch.empty(elems, dtype=torch.bfloat16, device=dev)
    a2a_in = torch.ones(elems, dtype=torch.bfloat16, device=dev); a2a_out = torch.empty_like(a2a_in)
    f = (world - 1) / world
    bench("all_reduce", lambda: dist.all_reduce(x), nbytes, 2 * f)
    bench("all_gather", lambda: dist.all_gather_into_tensor(out, shard), nbytes, f)
    bench("reduce_scatter", lambda: dist.reduce_scatter_tensor(shard, x), nbytes, f)
    bench("all_to_all", lambda: dist.all_to_all_single(a2a_out, a2a_in), nbytes, f)
    if rank == 0:
        print()
dist.destroy_process_group()
