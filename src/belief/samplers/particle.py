from __future__ import annotations

import dataclasses
import logging
import math
import random
from collections import abc as cabc

import numpy as np

import openspiel

# Type alias for opponent policy function: state -> action probabilities over full action space.
OpponentPolicy = cabc.Callable[[openspiel.State], np.ndarray]

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _StepRecord:
  """Record of one real move from the AI player's perspective."""

  actor_is_ai: bool
  ai_action: int | None
  ai_obs_after: bytes


class ParticleDeterminizationSampler:
  """Belief sampler using particle filtering with optional opponent policy guidance."""

  def __init__(
    self,
    game: openspiel.Game,
    ai_id: int,
    num_particles: int = 120,
    max_matching_opp_actions: int = 64,
    rebuild_max_tries: int = 200,
    seed: int = 0,
    opponent_policy: OpponentPolicy | None = None,
  ):
    if num_particles <= 0:
      raise ValueError("num_particles must be > 0")
    if max_matching_opp_actions <= 0:
      raise ValueError("max_matching_opp_actions must be > 0")
    if rebuild_max_tries <= 0:
      raise ValueError("rebuild_max_tries must be > 0")

    self.game = game
    self.ai_id = ai_id
    self.num_particles = num_particles
    self.max_matching_opp_actions = max_matching_opp_actions
    self.rebuild_max_tries = rebuild_max_tries
    self.rng = random.Random(seed)
    self.opponent_policy = opponent_policy

    self._history: list[_StepRecord] = []
    self._particles: dict[str, None] = {}  # using dict as an ordered set

  def get_particles(self, n: int) -> list[openspiel.State]:
    """Return up to n particles from the current belief."""
    particles: list[openspiel.State] = []
    if not self._particles:
      return particles
    for i in range(min(n, len(self._particles))):
      state = self.game.deserialize_state(list(self._particles)[i])
      particles.append(state)
    return particles

  def reset(self) -> None:
    """Clear history and particles for a new game."""
    self._history.clear()
    self._particles.clear()

  def step(
    self, actor: int, action: int, real_state_after: openspiel.State
  ) -> None:
    """Update belief after a real move.

    actor/action describe the real move taken in the environment.
    real_state_after is used ONLY to extract the AI observation (non-cheating).
    """
    rec = _StepRecord(
      actor_is_ai=(actor == self.ai_id),
      ai_action=(action if actor == self.ai_id else None),
      ai_obs_after=self._ai_obs(real_state_after),
    )
    self._history.append(rec)

    if not self._particles:
      self._rebuild_particles()
      return

    if rec.actor_is_ai:
      self._update_for_ai_move(rec)
    else:
      self._update_for_opponent_move(rec)

  def sample(self) -> openspiel.State:
    """Sample a particle consistent with current observation history.

    This function MUST NOT crash training. If belief collapses:
    - attempt rebuild
    - if still empty, fall back to a fresh initial state (uninformed determinization)
    """
    if not self._history:
      return self.game.new_initial_state()

    if not self._particles:
      self._rebuild_particles()

    if not self._particles:
      logger.warning(
        "Particle sampler belief collapsed with empty particles after rebuild; falling back to initial state."
      )
      return self.game.new_initial_state()

    logger.debug(
      "Particle sampler returning sample from %d particles.",
      len(self._particles),
    )
    return self.game.deserialize_state(self.rng.choice(list(self._particles)))

  def _ai_obs(self, state: openspiel.State) -> bytes:
    return np.asarray(
      state.observation_tensor(self.ai_id), dtype=np.float32
    ).tobytes()

  def _get_opponent_action_weights(
    self, state: openspiel.State
  ) -> list[float]:
    legal = state.legal_actions()
    if not legal:
      return []
    uniform_probs = np.ones(len(legal), dtype=np.float32) / len(legal)
    if self.opponent_policy is None:
      probs = uniform_probs
    else:
      policy_probs = self.opponent_policy(state)
      legal_policy_probs = np.array(
        [policy_probs[a] for a in legal], dtype=np.float32
      )
      legal_policy_probs = np.maximum(legal_policy_probs, 1e-12)
      legal_policy_probs /= legal_policy_probs.sum()

      probs = 0.5 * legal_policy_probs + 0.5 * uniform_probs

    return probs.tolist()

  def _update_for_ai_move(self, rec: _StepRecord) -> None:
    assert rec.actor_is_ai and rec.ai_action is not None
    updated: dict[str, None] = {}
    for p in self._particles:
      s = self.game.deserialize_state(p)
      if rec.ai_action not in s.legal_actions():
        continue
      s.apply_action(rec.ai_action)
      if self._ai_obs(s) == rec.ai_obs_after:
        updated[s.serialize()] = None
    self._particles = updated

  def _update_for_opponent_move(self, rec: _StepRecord) -> None:
    assert not rec.actor_is_ai
    candidates: list[tuple[str, float]] = []
    for p in self._particles:
      candidates.extend(self._matching_opponent_children(p, rec.ai_obs_after))

    self._particles = self._resample_unique_candidates(candidates)

  def _matching_opponent_children(
    self, particle: str, target_obs: bytes
  ) -> list[tuple[str, float]]:
    """Generate up to K matching opponent-successor states for a single particle.

    Returns list of (child_state, weight).
    Uses stochastic sampling from a mixture of policy and uniform to avoid collapse
    when the policy is wrong.
    """
    s = self.game.deserialize_state(particle)
    legal = s.legal_actions()
    if not legal:
      return []

    probs = self._get_opponent_action_weights(s)

    matches: list[tuple[str, float]] = []
    tried_actions: set[int] = set()

    while len(tried_actions) < len(legal):
      if len(matches) >= self.max_matching_opp_actions:
        break

      action_idx = self.rng.choices(range(len(legal)), weights=probs, k=1)[0]
      action = legal[action_idx]

      if action in tried_actions:
        continue
      tried_actions.add(action)

      p3 = self.game.deserialize_state(particle)
      p3.apply_action(action)
      if self._ai_obs(p3) != target_obs:
        continue

      weight = probs[action_idx]
      matches.append((p3.serialize(), weight))
    return matches

  def _resample_unique_candidates(
    self, candidates: list[tuple[str, float]]
  ) -> dict[str, None]:
    """Deduplicate candidates by serialize key, aggregate weights, then resample to num_particles."""
    if not candidates:
      return {}

    unique_weights: dict[str, float] = {}
    for k, w in candidates:
      if k in unique_weights:
        w0 = unique_weights[k]
        unique_weights[k] = w0 + float(w)
      else:
        unique_weights[k] = float(w)

    if len(unique_weights) <= self.num_particles:
      return dict.fromkeys(unique_weights)

    # Weighted sampling without replacement using Efraimidis–Spirakis keys:
    # For weight w, sample key u^(1/w). Larger key => higher chance.
    scored: list[tuple[float, str]] = []
    for k, w in unique_weights.items():
      if not math.isfinite(w) or w <= 0.0:
        w = 1e-12
      u = self.rng.random()
      u = max(u, 1e-12)
      score = u ** (1.0 / w)
      scored.append((score, k))

    scored.sort(reverse=True, key=lambda x: x[0])
    chosen = scored[: self.num_particles]
    return {k: None for _, k in chosen}

  def _rebuild_particles(self) -> None:
    """Rebuild particles from scratch using the stored observation history."""
    if not self._history:
      self._particles = {}
      return

    # Preserve current K and optionally escalate on later attempts.
    base_k = self.max_matching_opp_actions

    for attempt in range(self.rebuild_max_tries):
      particles: dict[str, None] = {}
      s0 = self.game.new_initial_state()
      particles[s0.serialize()] = None

      # Mild excalation on later attempts to increase diversity if rebuild keeps failing.
      self.max_matching_opp_actions = min(
        base_k + attempt, self.game.num_distinct_actions()
      )

      ok = True
      for i, rec in enumerate(self._history):
        if not particles:
          ok = False
          break

        if rec.actor_is_ai:
          updated: dict[str, None] = {}
          assert rec.ai_action is not None
          for p in particles:
            p2 = self.game.deserialize_state(p)
            if rec.ai_action not in p2.legal_actions():
              continue
            p2.apply_action(rec.ai_action)
            if self._ai_obs(p2) == rec.ai_obs_after:
              updated[p2.serialize()] = None
          particles = updated
        else:
          candidates: list[tuple[str, float]] = []
          for p in particles:
            candidates.extend(
              self._matching_opponent_children(p, rec.ai_obs_after)
            )
          particles = self._resample_unique_candidates(candidates)

      if ok and particles:
        self._particles = particles
        self.max_matching_opp_actions = base_k
        return

    # Rebuild failed.
    self._particles = {}
    self.max_matching_opp_actions = base_k
