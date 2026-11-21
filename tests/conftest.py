"""conftest.py — project-wide pytest fixtures"""
import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fixed_seed():
    """Set random seed to 42 before every test. This is the project-wide default."""
    random.seed(42)
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass
    yield
