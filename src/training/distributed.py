import os
import torch
import torch.distributed as dist

def init_distributed():
    ddp = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
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
    Average a scalar across all DDP ranks. If not in DDP, return x.
    Used in the trainer to get the average train and val loss across ranks.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t.item() / dist.get_world_size())