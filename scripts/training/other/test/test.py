import torch, json
from pathlib import Path
# import your modules as in train
from src.training.trainer import test
from src.models.custom_transformer import YearProcessor
from src.training.stats import load_and_filter_standardisation
from src.dataset.dataloader import get_train_val_test, get_data
from src.dataset.variables import *
import os, sys, json, argparse, time, logging, random, math
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tempfile, shutil, signal

# ------------------------------ Environment Setup -----------------------------------
# set project root
project_root = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel")
sys.path.append(str(project_root))

training_dir = project_root / "scripts/training"

# start time
start_dt = datetime.now() 
start_dt_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

# import paths and variables
from src.paths.paths import *
from src.dataset.variables import *

# import training modules
from src.models.custom_transformer import YearProcessor
from torch.utils.data import Dataset, random_split, Subset
from src.training.checkpoints import save_cb
from src.training.history import History
from src.training.loss import custom_loss
from src.training.trainer import fit, plan_validation, unwrap, test 
from src.dataset.dataloader import get_train_val_test, get_data
from src.training.distributed import  init_distributed
from src.training.logging import save_args, setup_logging 
from src.training.scheduler import build_cosine_wr_scheduler
from src.training.stats import get_split_stats, set_seed, load_and_filter_standardisation
from src.dataset.dataset import base, get_subset

# Slurm stuff
workers = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "4")))
slurm_id = os.getenv("SLURM_JOB_ID", "no_slurm_id")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# rebuild datasets & loaders (small workers, no pin)
std_dict, pruned = load_and_filter_standardisation(
    standardisation_path=std_dict_path,
    all_vars=all_vars,
    daily_vars=daily_vars,
    monthly_vars=monthly_vars,
    annual_vars=annual_vars,
    monthly_states=monthly_states,
    annual_states=annual_states,
    exclude_vars=set(),
)
ds = get_train_val_test(std_dict)
train_ds, val_ds, test_ds = ds["train"], ds["val"], ds["test"]
_, _, test_dl = get_data(train_ds, val_ds, test_ds, bs=1, num_workers=2, ddp=False)

# build model dims like in training
monthly_list = pruned['monthly_fluxes'] + pruned['monthly_states']
annual_list  = pruned['annual_states']
output_dim = len(monthly_list + annual_list)
input_dim = (
    len(pruned['daily_forcing']) + len(pruned['monthly_forcing']) +
    len(pruned['monthly_states']) + len(pruned['annual_forcing']) +
    len(pruned['annual_states'])
)

model = YearProcessor(
    input_dim=input_dim, output_dim=output_dim,
    d=128, h=1024, g=256, num_layers=4, nhead=8, dropout=0.1,
    transformer_kwargs={"max_len":31}
).float().to(DEVICE)
model.eval()

best = Path("/Net/Groups/BGI/people/ecathain/TRENDY_Emulator_Scripts/NewModel/scripts/training/runs/2025-09-24/3525650_full_run/saves/best_weights.pt")
state = torch.load(best, map_location="cpu", weights_only=True)
model.load_state_dict(state)

from src.training.loss import custom_loss
loss_fn = custom_loss(list(range(len(monthly_list))),
                      list(range(len(monthly_list), output_dim)),
                      loss_type="mse", reduction="mean")

with torch.no_grad():
    out = test(model, loss_fn, test_dl, DEVICE, logger=None)
global_avg = out["sum_loss"]/max(1.0, out["count"])
print(json.dumps({"test_avg_loss": float(global_avg)}, indent=2))