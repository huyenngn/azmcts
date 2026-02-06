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
  ai_obs_after: str


class ParticleDeterminizationSampler:
  """Belief sampler using particle filtering with optional opponent policy guidance."""

  def __init__(
    self,
    game: openspiel.Game,
    ai_id: int,
    num_particles: int = 24,
    max_matching_opp_actions: int = 24,
    min_matching_opp_actions: int = 2,
    rebuild_max_tries: int = 200,
    matching_tries_per_particle: int = 100,
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
    self.min_matching_opp_actions = min_matching_opp_actions
    self.rebuild_max_tries = rebuild_max_tries
    self.matching_tries_per_particle = matching_tries_per_particle
    self.rng = random.Random(seed)
    self.opponent_policy = opponent_policy

    self._history: list[_StepRecord] = []
    self._particles: dict[str, openspiel.State] = {}

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
    return self.rng.choice(list(self._particles.values())).clone()

  def _ai_obs(self, state: openspiel.State) -> str:
    return state.observation_string(self.ai_id)

  def _get_opponent_action_weights(
    self, state: openspiel.State
  ) -> list[float]:
    legal = state.legal_actions()
    if not legal:
      return []
    uniform_probs = np.ones(len(legal), dtype=np.float64) / len(legal)
    if self.opponent_policy is None:
      probs = uniform_probs
    else:
      policy_probs = self.opponent_policy(state)
      legal_policy_probs = np.array(
        [policy_probs[a] for a in legal], dtype=np.float64
      )
      legal_policy_probs = np.maximum(legal_policy_probs, 1e-12)
      legal_policy_probs /= legal_policy_probs.sum()

      probs = 0.5 * legal_policy_probs + 0.5 * uniform_probs

    return probs.tolist()

  def _update_for_ai_move(self, rec: _StepRecord) -> None:
    assert rec.actor_is_ai and rec.ai_action is not None
    updated: dict[str, openspiel.State] = {}
    for p in self._particles.values():
      p2 = p.clone()
      if rec.ai_action not in p2.legal_actions():
        continue
      p2.apply_action(rec.ai_action)
      if self._ai_obs(p2) == rec.ai_obs_after:
        updated[p2.serialize()] = p2
    self._particles = updated

  def _update_for_opponent_move(self, rec: _StepRecord) -> None:
    assert not rec.actor_is_ai
    candidates: list[tuple[openspiel.State, float]] = []
    for p in self._particles.values():
      candidates.extend(self._matching_opponent_children(p, rec.ai_obs_after))

    self._particles = self._resample_unique_candidates(candidates)
    self._downsample_particles_uniform()

  def _matching_opponent_children(
    self, particle: openspiel.State, target_obs: str
  ) -> list[tuple[openspiel.State, float]]:
    """Generate up to K matching opponent-successor states for a single particle.

    Returns list of (child_state, weight).
    Uses stochastic sampling from a mixture of policy and uniform to avoid collapse
    when the policy is wrong.
    """
    p2 = particle.clone()
    legal = p2.legal_actions()
    if not legal:
      return []

    probs = self._get_opponent_action_weights(p2)

    matches: list[tuple[openspiel.State, float]] = []
    tried_actions: set[int] = set()

    for _ in range(self.matching_tries_per_particle):
      if len(matches) >= self.max_matching_opp_actions:
        break
      if len(tried_actions) >= len(legal):
        break

      action_idx = self.rng.choices(range(len(legal)), weights=probs, k=1)[0]
      action = legal[action_idx]

      if action in tried_actions:
        continue
      tried_actions.add(action)

      p3 = p2.clone()
      p3.apply_action(action)
      if self._ai_obs(p3) != target_obs:
        continue

      weight = probs[action_idx]
      matches.append((p3, weight))

    return matches

  def _downsample_particles_uniform(self) -> None:
    """Uniformly downsample current unique particles to num_particles (if needed)."""
    if len(self._particles) <= self.num_particles:
      return
    keys = list(self._particles.keys())
    keep = set(self.rng.sample(keys, self.num_particles))
    self._particles = {k: self._particles[k] for k in keep}

  def _resample_unique_candidates(
    self, candidates: list[tuple[openspiel.State, float]]
  ) -> dict[str, openspiel.State]:
    """Deduplicate candidates by serialize key, aggregate weights, then resample to num_particles."""
    if not candidates:
      return {}

    by_key: dict[str, tuple[openspiel.State, float]] = {}
    for s, w in candidates:
      k = s.serialize()
      if k in by_key:
        rep, w0 = by_key[k]
        by_key[k] = (rep, w0 + float(w))
      else:
        by_key[k] = (s, float(w))

    items = list(by_key.items())
    if len(items) <= self.num_particles:
      return {k: st for k, (st, _) in items}

    # Weighted sampling without replacement using Efraimidis–Spirakis keys:
    # For weight w, sample key u^(1/w). Larger key => higher chance.
    scored: list[tuple[float, str, openspiel.State]] = []
    for k, (st, w) in items:
      if not math.isfinite(w) or w <= 0.0:
        w = 1e-12
      u = self.rng.random()
      u = max(u, 1e-12)
      score = u ** (1.0 / w)
      scored.append((score, k, st))

    scored.sort(reverse=True, key=lambda x: x[0])
    chosen = scored[: self.num_particles]
    return {k: st for _, k, st in chosen}

  def _rebuild_particles(self) -> None:
    """Rebuild particles from scratch using the stored observation history."""
    if not self._history:
      self._particles = {}
      return

    # Preserve current K and optionally escalate on later attempts.
    base_k = self.max_matching_opp_actions

    for attempt in range(self.rebuild_max_tries):
      particles: dict[str, openspiel.State] = {}
      s0 = self.game.new_initial_state()
      particles[s0.serialize()] = s0

      # Mild excalation on later attempts to increase diversity if rebuild keeps failing.
      attempt_max_opp_actions = min(
        base_k + attempt, self.game.num_distinct_actions()
      )

      ok = True
      for i, rec in enumerate(self._history):
        if not particles:
          ok = False
          break

        # Aggressive decay as game progresses
        self.max_matching_opp_actions = max(
          int(attempt_max_opp_actions / ((i + 1) ** 2)),
          self.min_matching_opp_actions,
        )

        if rec.actor_is_ai:
          updated: dict[str, openspiel.State] = {}
          assert rec.ai_action is not None
          for p in particles.values():
            p2 = p.clone()
            if rec.ai_action not in p2.legal_actions():
              continue
            p2.apply_action(rec.ai_action)
            if self._ai_obs(p2) == rec.ai_obs_after:
              updated[p2.serialize()] = p2
          particles = updated
        else:
          candidates: list[tuple[openspiel.State, float]] = []
          for p in particles.values():
            candidates.extend(
              self._matching_opponent_children(p, rec.ai_obs_after)
            )
          particles = self._resample_unique_candidates(candidates)

      if ok and particles:
        self._particles = particles
        self.max_matching_opp_actions = base_k
        self._downsample_particles_uniform()
        return

    # Rebuild failed.
    self._particles = {}
    self.max_matching_opp_actions = base_k
