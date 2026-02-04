from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
  import random

  import openspiel


class DeterminizationSampler(t.Protocol):
  """Protocol for samplers that generate determinizations from belief states."""

  def sample(
    self, state: openspiel.State, rng: random.Random
  ) -> openspiel.State: ...
