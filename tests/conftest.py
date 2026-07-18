import sys
from pathlib import Path

import pytest

# make the example model importable by the tests (it doubles as a fixture)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: needs a CUDA device (larger-chi oracle tests)")


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if not has_cuda:
        skip = pytest.mark.skip(reason="no CUDA device available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
