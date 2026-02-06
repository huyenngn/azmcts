from __future__ import annotations

import dataclasses
import logging
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
    min_particles: int = 32,
    max_matching_opp_actions: int = 8,
    rebuild_max_tries: int = 200,
    seed: int = 0,
    opponent_policy: OpponentPolicy | None = None,
  ):
    if min_particles <= 0:
      raise ValueError("min_particles must be > 0")
    if max_matching_opp_actions <= 0:
      raise ValueError("max_matching_opp_actions must be > 0")
    if rebuild_max_tries <= 0:
      raise ValueError("rebuild_max_tries must be > 0")

    self.game = game
    self.ai_id = ai_id
    self.min_particles = min_particles
    self.max_matching_opp_actions = max_matching_opp_actions
    self.rebuild_max_tries = rebuild_max_tries
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

    return self.rng.choice(list(self._particles.values())).clone()

  def _ai_obs(self, state: openspiel.State) -> str:
    return state.observation_string(self.ai_id)

  def _get_opponent_actions_and_weights(
    self, state: openspiel.State
  ) -> list[tuple[int, float]]:
    """Get legal opponent actions in a deterministic order (policy) or random permutation (no policy)."""
    legal = state.legal_actions()
    if not legal:
      return []

    if self.opponent_policy is None:
      return [(a, 1.0) for a in self.rng.sample(legal, len(legal))]

    probs = self.opponent_policy(state)
    legal_probs = np.array([probs[a] for a in legal], dtype=np.float64)
    legal_probs = np.maximum(legal_probs, 1e-12)
    total = float(legal_probs.sum())
    if total <= 0.0:
      return [(a, 1.0) for a in self.rng.sample(legal, len(legal))]

    legal_probs /= total
    return sorted(zip(legal, legal_probs), key=lambda x: x[1], reverse=True)

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

  def _matching_opponent_children(
    self, particle: openspiel.State, target_obs: str
  ) -> list[tuple[openspiel.State, float]]:
    """Generate up to K matching opponent-successor states for a single particle.

    Returns list of (child_state, weight).
    Weight is proportional to opponent policy prob for the chosen action within legal actions.
    """
    p2 = particle.clone()
    if not p2.legal_actions():
      return []

    opp_actions_and_weights = self._get_opponent_actions_and_weights(p2)
    if not opp_actions_and_weights:
      return []

    matches: list[tuple[openspiel.State, float]] = []
    seen: set[int] = set()

    for a, w in opp_actions_and_weights:
      if a in seen:
        continue
      seen.add(a)

      p3 = p2.clone()
      p3.apply_action(a)
      if self._ai_obs(p3) != target_obs:
        continue

      matches.append((p3, w))
      if len(matches) >= self.max_matching_opp_actions:
        break

    return matches

  def _resample_unique_candidates(
    self, candidates: list[tuple[openspiel.State, float]]
  ) -> dict[str, openspiel.State]:
    """Deduplicate candidates by serialize key."""
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

    return {k: st for k, (st, _) in by_key.items()}

  def _rebuild_particles(self) -> None:
    """Rebuild particles from scratch using the stored observation history."""
    if not self._history:
      self._particles = {}
      return

    # Preserve current K and optionally escalate on later attempts.
    base_k = self.max_matching_opp_actions

    for attempt in range(self.rebuild_max_tries):
      # Mild escalation every few attempts if we're repeatedly failing.
      if attempt > 0 and attempt % 10 == 0:
        self.max_matching_opp_actions = min(base_k * 2, 64)

      attempt_k = self.max_matching_opp_actions

      particles: dict[str, openspiel.State] = {}
      s0 = self.game.new_initial_state()
      particles[s0.serialize()] = s0

      ok = True
      for rec in self._history:
        if not particles:
          ok = False
          break

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

        if len(particles) > self.min_particles:
          self.max_matching_opp_actions = 1
        else:
          self.max_matching_opp_actions = attempt_k

      if ok and particles:
        self._particles = particles
        self.max_matching_opp_actions = base_k
        return

    # Rebuild failed.
    self._particles = {}
    self.max_matching_opp_actions = base_k
