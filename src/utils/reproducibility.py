"""
Reproducibility utilities for seeding and versioning.
"""
import random
import numpy as np
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Try to import torch, but don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
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
    info = {
        "python_version": sys.version,
    }
    
    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False
    
    return info
