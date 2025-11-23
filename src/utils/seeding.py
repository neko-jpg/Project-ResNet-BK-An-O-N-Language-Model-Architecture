import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set seed for reproducibility across Python, NumPy, and PyTorch.
    Logs the seed used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Global seed set to: {seed}")
    print(f"Global seed set to: {seed}")
