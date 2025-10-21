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
        # Initialize the default process group. NCCL is the standard backend
        # for multi-GPU training on Linux (GPU-only).
        dist.init_process_group(backend="nccl")

        # LOCAL_RANK is set by torchrun/torch.distributed.launch
        local_rank = int(os.environ["LOCAL_RANK"])

        # Bind this process to its GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Query group metadata
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        # Single-process fallback
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