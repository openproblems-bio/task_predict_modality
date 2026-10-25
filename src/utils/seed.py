import os
import random
import sys

import numpy as np


def seed_everything(seed):
    """Seed python's and numpy's random number generators, and those of torch and tensorflow when the method has
    imported them (so call this after the imports), and make them pick deterministic GPU algorithms, so that a method
    gives the same predictions every time it runs on the same data."""
    random.seed(seed)
    np.random.seed(seed)
    # cuBLAS is only deterministic with a fixed workspace, read when CUDA initialises
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)  # seeds every CUDA device too
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn rather than fail on the few operations that have no deterministic implementation
        torch.use_deterministic_algorithms(True, warn_only=True)
    if "tensorflow" in sys.modules:
        tensorflow = sys.modules["tensorflow"]
        tensorflow.keras.utils.set_random_seed(seed)
        tensorflow.config.experimental.enable_op_determinism()
