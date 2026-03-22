import random

import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
