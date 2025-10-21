from pathlib import Path
from typing import Optional
import sys
import logging
import json

def setup_logging(out_dir: Path, job_name: str, log_filename: Optional[str] = None) -> logging.Logger:
    """
    Configure a file+stdout logger under `out_dir/logs/`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(job_name)
    log.setLevel(logging.INFO)

    # Clear pre-existing handlers to avoid duplicate logs when re-running
    if log.handlers:
        for h in list(log.handlers):
            log.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Stream handler (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    # File handler
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    lf = log_filename if log_filename is not None else f"{job_name}.log"
    fh = logging.FileHandler(logs_dir / lf)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    log.addHandler(sh)
    log.addHandler(fh)
    return log

def save_args(run_dir, args):
    info_dir = Path(run_dir) / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    try:
        import yaml
        out_path = info_dir / "args.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(args_dict, f, sort_keys=False, default_flow_style=False)
    except ImportError:
        out_path = info_dir / "args.json"
        with open(out_path, "w") as f:
            json.dump(args_dict, f, indent=2)
    return out_path




