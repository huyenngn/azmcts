from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    import random

    import pyspiel


class DeterminizationSampler(t.Protocol):
    """Protocol for samplers that generate determinizations from belief states."""

    def sample(
        self, state: pyspiel.State, rng: random.Random
    ) -> pyspiel.State: ...
