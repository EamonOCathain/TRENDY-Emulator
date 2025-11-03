import os
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Distributed initialization & simple reductions
# ---------------------------------------------------------------------------

def init_distributed():
    """
    Initialize (or bypass) PyTorch Distributed Data Parallel (DDP).

    Behavior:
      - If environment variables 'RANK' and 'WORLD_SIZE' are present, we assume
        the process is launched under a DDP launcher (e.g., torchrun) and:
          * initialize a process group (NCCL backend),
          * set the CUDA device to LOCAL_RANK,
          * return device, local rank, world size, and rank from the group.
      - Otherwise, we run in single-process mode:
          * pick 'cuda' if available else 'cpu',
          * world_size=1, rank=0, local_rank=0.

    Returns:
      ddp (bool):          True if DDP is initialized, else False.
      device (torch.device): Current device for this process.
      local_rank (int):    LOCAL_RANK in DDP or 0 in single-process.
      world_size (int):    Number of processes in the group (1 if non-DDP).
      rank (int):          Rank of this process in the group (0 if non-DDP).
    """
    ddp = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # 1) Bind this process to its GPU *first*
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # 2) Now initialize the process group
        dist.init_process_group(backend="nccl", init_method="env://")

        # 3) Touch the device so NCCL sees the CUDA context
        _ = torch.empty(0, device=device)

        # 4) Device-aware barrier (prevents the NCCL warning/hang risk)
        dist.barrier(device_ids=[local_rank])

        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0

    return ddp, device, local_rank, world_size, rank


def ddp_mean_scalar(x: float, device: torch.device) -> float:
    """
    Average a scalar value across all DDP ranks.

    - In non-DDP or uninitialized contexts, returns the input unchanged.
    - In DDP, performs an all-reduce (SUM) and divides by world size.

    Args:
      x:      The scalar (Python float) to average across ranks.
      device: Device to place the temporary tensor on (e.g., model's device).

    Returns:
      float: The mean value of `x` across all ranks (or `x` if not in DDP).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x

    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()
  
def cleanup_distrib_and_cuda(ddp: bool):
    """
    Best-effort shutdown to avoid rank hangs at script end.
    Must be called by *every* rank (not just rank 0).
    """
    # Drain CUDA work (safe even on CPU-only)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    # Try to close any dl workers (best-effort; they may be already gone)
    try:
        for dl in ("train_dl", "valid_dl", "test_dl"):
            if dl in globals():
                obj = globals()[dl]
                # Torch DataLoader cleanup knobs (no-op if not available)
                try:
                    if hasattr(obj, "_iterator") and obj._iterator is not None:
                        obj._iterator._shutdown_workers()  # type: ignore[attr-defined]
                except Exception:
                    pass
    except Exception:
        pass

    # Last barrier so all ranks reach the same point
    try:
        if ddp and dist.is_available() and dist.is_initialized():
            # Using a barrier without device_ids is safer across backends
            dist.barrier()
    except Exception:
        pass

    # Destroy process group
    try:
        if ddp and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    # Free big tensors/models and empty CUDA cache
    try:
        if "model" in globals():
            del globals()["model"]
        if "real_model" in globals():
            del globals()["real_model"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Force GC to break ref cycles from workers/queues
    try:
        gc.collect()
    except Exception:
        pass