import torch
import random
import numpy as np


def set_deterministic(seed: int = 123456789):
    print('INFO: setting random seed to {}'.format(seed))
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
