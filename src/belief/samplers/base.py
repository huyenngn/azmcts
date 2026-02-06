from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
  import openspiel


class DeterminizationSampler(t.Protocol):
  """Protocol for samplers that generate determinizations from belief states."""

  _particles: dict[str, None]

  def sample(self) -> openspiel.State: ...
