from __future__ import annotations

import dataclasses
import random
import typing as t

import numpy as np

if t.TYPE_CHECKING:
  import openspiel


class Agent(t.Protocol):
  """Protocol for game-playing agents."""

  def select_action(self, state: openspiel.State) -> int:
    """Select an action given the current game state."""
    ...


@dataclasses.dataclass
class AgentConfig:
  """Configuration for agent initialization."""

  seed: int = 0


class BaseAgent:
  """Base class providing RNG and observation utilities."""

  def __init__(self, player_id: int, num_actions: int, seed: int = 0):
    self.player_id = player_id
    self.num_actions = num_actions
    self.rng = random.Random(seed)

  def obs_key(self, state: openspiel.State, player_id: int) -> str:
    """Return observation string for the specified player."""
    return state.observation_string(player_id)

  def obs_tensor(self, state: openspiel.State, player_id: int) -> np.ndarray:
    """Return observation tensor for the specified player."""
    return np.asarray(state.observation_tensor(player_id), dtype=np.float32)


class PolicyTargetMixin:
  """Mixin for agents that provide policy targets for training."""

  def select_action_with_pi(
    self, state: openspiel.State, temperature: float = 1.0
  ) -> tuple[int, np.ndarray]:
    raise NotImplementedError
