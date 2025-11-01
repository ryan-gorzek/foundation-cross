"""
Reproducibility utilities for seeding and versioning.
"""
import random
import numpy as np
import torch
import subprocess
from pathlib import Path
from typing import Optional


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_status() -> Optional[str]:
    """Check if git repo has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent
        )
        status = result.stdout.strip()
        return "clean" if not status else "dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_environment_info() -> dict:
    """Get environment information for reproducibility."""
    import torch
    import sys
    
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_name"] = torch.cuda.get_device_name(0)
    
    return info