"""Centralized seed derivation for reproducibility.

All stochastic operations should derive their seeds from a base_seed via
derive_seed() with appropriate namespacing to ensure:
  - Same base_seed + same arguments => same derived seed
  - Different purpose/run_id/game_idx/player_id => different derived seed

Seed Namespaces (purpose strings):
  - train/selfplay    : Self-play game generation
  - train/sgd         : SGD batch shuffling
  - train/agent       : Per-agent RNG in training
  - train/belief      : Belief sampler in training
  - eval/match        : Evaluation match loop
  - eval/agent        : Per-agent RNG in evaluation
  - eval/belief       : Belief sampler in evaluation
  - eval/rng          : Random baseline in evaluation
  - api/rng           : API backend random fallback
  - api/agent         : Per-agent RNG in API
  - api/belief        : Belief sampler in API

Usage:
    seed = derive_seed(base_seed, purpose="eval/agent", game_idx=i, player_id=0)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import random
import subprocess
import sys
import typing as t

import numpy as np
import torch

_UINT32_MASK = 0xFFFFFFFF


def _mix_to_u32(s: str) -> int:
    """Hash string to 32-bit unsigned integer via blake2b."""
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") & _UINT32_MASK


def derive_seed(
    base_seed: int,
    *,
    purpose: str,
    run_id: str = "",
    step: int | None = None,
    game_idx: int | None = None,
    player_id: int | None = None,
    extra: str = "",
) -> int:
    """Deterministic seed derivation with namespacing.

    Uses blake2b hashing to mix all parameters into a 32-bit seed.
    Same inputs -> same seed; different purpose/ids -> different seed.

    Args:
        base_seed: Root seed for the entire run.
        purpose: Namespace string (e.g., "train/selfplay", "eval/agent").
        run_id: Unique identifier for the run (e.g., run directory name).
        step: Training step or iteration number.
        game_idx: Game index within a match or self-play batch.
        player_id: Player identifier (0 or 1).
        extra: Additional differentiator string.

    Returns
    -------
        32-bit unsigned integer seed.
    """
    key = (
        f"{base_seed}|{purpose}|{run_id}|{step}|{game_idx}|{player_id}|{extra}"
    )
    return _mix_to_u32(key)


def _get_git_commit() -> str | None:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def _get_git_dirty() -> bool | None:
    """Check if git working tree is dirty, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except Exception:
        pass
    return None


@dataclasses.dataclass
class ReproFingerprint:
    """Environment fingerprint for reproducibility tracking."""

    python_version: str
    platform: str
    torch_version: str | None
    cuda_version: str | None
    cudnn_version: str | None
    device_name: str | None
    git_commit: str | None
    git_dirty: bool | None

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "device_name": self.device_name,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def get_repro_fingerprint(device: str = "cpu") -> ReproFingerprint:
    """Collect environment fingerprint for reproducibility logging.

    Args:
        device: PyTorch device string (e.g., "cpu", "cuda:0").

    Returns
    -------
        ReproFingerprint with system and library versions.
    """
    torch_version: str | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    device_name: str | None = None

    try:
        torch_version = torch.__version__
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            cudnn_version = str(torch.backends.cudnn.version())
            if device.startswith("cuda"):
                idx = 0
                if ":" in device:
                    idx = int(device.split(":")[1])
                device_name = torch.cuda.get_device_name(idx)
    except ImportError:
        pass

    return ReproFingerprint(
        python_version=sys.version,
        platform=platform.platform(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        device_name=device_name,
        git_commit=_get_git_commit(),
        git_dirty=_get_git_dirty(),
    )


def log_repro_fingerprint(device: str = "cpu") -> None:
    """Print reproducibility fingerprint to stdout."""
    fp = get_repro_fingerprint(device)
    print("=== Reproducibility Fingerprint ===")
    print(f"  Python: {fp.python_version.split()[0]}")
    print(f"  Platform: {fp.platform}")
    if fp.torch_version:
        print(f"  PyTorch: {fp.torch_version}")
    if fp.cuda_version:
        print(f"  CUDA: {fp.cuda_version}")
    if fp.cudnn_version:
        print(f"  cuDNN: {fp.cudnn_version}")
    if fp.device_name:
        print(f"  Device: {fp.device_name}")
    if fp.git_commit:
        dirty_str = " (dirty)" if fp.git_dirty else ""
        print(f"  Git: {fp.git_commit}{dirty_str}")
    print("===================================")


def set_global_seeds(
    seed: int,
    deterministic_torch: bool = False,
    log: bool = True,
) -> None:
    """Set global random seeds for reproducibility.

    This sets seeds for:
      - random.seed(seed)
      - numpy.random.seed(seed) [for legacy code only]
      - torch.manual_seed(seed)
      - torch.cuda.manual_seed_all(seed)

    If deterministic_torch is True, also enables:
      - torch.backends.cudnn.deterministic = True
      - torch.backends.cudnn.benchmark = False
      - torch.use_deterministic_algorithms(True)

    Args:
        seed: Base seed for all RNGs.
        deterministic_torch: Enable extra determinism flags (may hurt performance).
        log: Print a summary of what was set.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # Not available in older PyTorch versions

    if log:
        det_str = "ON" if deterministic_torch else "OFF"
        print(f"[seeding] base_seed={seed} deterministic_torch={det_str}")
