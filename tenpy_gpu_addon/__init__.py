"""GPU acceleration for TeNPy DMRG — no changes to TeNPy required.

Three entry points, all consuming stock tenpy objects through the public API:

- `run_dmrg_torch(model, ...)` — fully GPU-resident two-site DMRG loop
  (fastest; finite, charge-free models).
- `TorchTwoSiteDMRGEngine` — drop-in replacement for tenpy's
  `TwoSiteDMRGEngine` with GPU-resident Lanczos (keeps every tenpy feature).
- `DeviceMPS` — tenpy-API-compatible GPU measurement (expectation values,
  correlation functions, overlaps, entropies) by duck typing.
- `TorchTDVP` — two-site TDVP real/imaginary time evolution on the GPU,
  sharing the DMRG loop's environment and effective-Hamiltonian kernels.

Both GPU paths fall back to CPU when CUDA is absent.
"""
__version__ = "0.1.0"

from .dense import export_mpo, export_mps, import_mps
from .dmrg import run_dmrg_torch
from .measure import DeviceMPS
from .engine import TorchTwoSiteDMRGEngine
from .tdvp import TorchTDVP

__all__ = [
    "run_dmrg_torch",
    "TorchTwoSiteDMRGEngine",
    "DeviceMPS",
    "TorchTDVP",
    "export_mpo",
    "export_mps",
    "import_mps",
]
