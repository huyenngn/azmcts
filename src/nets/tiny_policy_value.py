from __future__ import annotations

import dataclasses
import pathlib

import torch
from torch import nn
from torch.nn import functional as F


class TinyPolicyValueNet(nn.Module):
    """Minimal policy/value network."""

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256):
        """Initialize network layers."""
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions

        self.fc1 = nn.Linear(obs_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (policy_logits, value)."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy(x)
        v = torch.tanh(self.value(x))
        return logits, v


@dataclasses.dataclass
class _AZShared:
    net: TinyPolicyValueNet | None = None
    device: str | None = None
    path: str | None = None
    obs_size: int | None = None
    num_actions: int | None = None


_AZ_SHARED: _AZShared = _AZShared(
    net=None,
    device=None,
    path=None,
    obs_size=None,
    num_actions=None,
)


def get_shared_az_model(
    obs_size: int,
    num_actions: int,
    model_path: str,
    device: str = "cpu",
) -> TinyPolicyValueNet:
    """Load or create a shared model instance.

    Args:
        obs_size: Observation size for the network.
        num_actions: Number of actions for the network.
        model_path: Path to the model weights file.
        device: Device to load the model on (default: "cpu").

    Returns
    -------
        A shared TinyPolicyValueNet instance.
    """
    if (
        _AZ_SHARED.net is not None
        and _AZ_SHARED.device == device
        and _AZ_SHARED.path == model_path
        and _AZ_SHARED.obs_size == obs_size
        and _AZ_SHARED.num_actions == num_actions
    ):
        return _AZ_SHARED.net

    net = TinyPolicyValueNet(obs_size=obs_size, num_actions=num_actions).to(
        device
    )
    net.eval()

    p = pathlib.Path(model_path)
    if p.exists():
        state_dict = torch.load(str(p), map_location=device)
        net.load_state_dict(state_dict)

    _AZ_SHARED.net = net
    _AZ_SHARED.device = device
    _AZ_SHARED.path = model_path
    _AZ_SHARED.obs_size = obs_size
    _AZ_SHARED.num_actions = num_actions
    return net
