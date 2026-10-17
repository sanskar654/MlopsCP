"""
Seed & Reproducibility Utilities
================================
Sets deterministic seeds across Python, NumPy, PyTorch, and Scikit-Learn.
"""

import os
import random
import numpy as np

def fix_all_seeds(seed: int = 42) -> None:
    """Set global random seeds for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
