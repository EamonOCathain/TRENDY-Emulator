from pathlib import Path
from typing import Optional
import sys
import logging
import json


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def setup_logging(out_dir: Path, job_name: str, log_filename: Optional[str] = None) -> logging.Logger:
    """
    Configure a logger that writes both to stdout and to a file under `out_dir/logs/`.

    Behavior:
      - Ensures `out_dir/logs/` exists.
      - Clears any existing handlers for this logger (avoids duplicate lines).
      - Writes INFO-level messages to both console and log file.

    Args:
      out_dir (Path):   Directory under which the log folder will be created.
      job_name (str):   Name for the logger and log file (used as prefix).
      log_filename (Optional[str]): Override log filename (default: '{job_name}.log').

    Returns:
      logging.Logger: Configured logger ready for use.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(job_name)
    log.setLevel(logging.INFO)

    # Remove pre-existing handlers to prevent duplication when re-running scripts
    if log.handlers:
        for h in list(log.handlers):
            log.removeHandler(h)

    # Common message format
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # ---- Stream handler (stdout) ----
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    # ---- File handler ----
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    lf = log_filename if log_filename is not None else f"{job_name}.log"
    fh = logging.FileHandler(logs_dir / lf)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    # Register both handlers
    log.addHandler(sh)
    log.addHandler(fh)

    return log


# ---------------------------------------------------------------------------
# Argument saving utility
# ---------------------------------------------------------------------------

def save_args(run_dir: Path, args) -> Path:
    """
    Save CLI args or configuration namespace as YAML (or JSON fallback)
    into `run_dir/info/args.yaml`.

    Behavior:
      - Converts Path objects to strings for safe serialization.
      - Prefers YAML output if `pyyaml` is available, otherwise JSON.

    Args:
      run_dir (Path): Target run directory (creates `info/` subfolder).
      args:           Namespace or dataclass with `. __dict__` available.

    Returns:
      Path: Path to the saved argument file.
    """
    info_dir = Path(run_dir) / "info"
    info_dir.mkdir(parents=True, exist_ok=True)

    # Convert args to a JSON/YAML-safe dict
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