from __future__ import annotations

import dataclasses
import random
from collections import abc as cabc

import numpy as np

import openspiel

# Type alias for opponent policy function: state -> action probabilities
OpponentPolicy = cabc.Callable[[openspiel.State], np.ndarray]


@dataclasses.dataclass
class _StepRecord:
  """Record of a single game step for belief reconstruction."""

  actor_is_ai: bool
  ai_action: int | None
  ai_obs_after: str


class ParticleBeliefSampler:
  """Non-cheating belief sampler using particle filtering.

  Maintains particles (fully-determined states) consistent with
  the AI player's observation history without using hidden information.
  """

  def __init__(
    self,
    game: openspiel.Game,
    ai_id: int,
    num_particles: int = 32,
    opp_tries_per_particle: int = 8,
    rebuild_max_tries: int = 200,
    seed: int = 0,
    opponent_policy: OpponentPolicy | None = None,
  ):
    self.game = game
    self.ai_id = ai_id
    self.num_particles = num_particles
    self.opp_tries_per_particle = opp_tries_per_particle
    self.rebuild_max_tries = rebuild_max_tries
    self.rng = random.Random(seed)
    self.opponent_policy = opponent_policy

    self._history: list[_StepRecord] = []
    self._particles: list[openspiel.State] = []

  def _sample_opponent_action(self, state: openspiel.State) -> int:
    """Sample an opponent action, using policy if available."""
    legal = state.legal_actions()
    if not legal:
      raise ValueError("No legal actions for opponent sampling")

    if self.opponent_policy is None:
      return self.rng.choice(legal)

    # Sample from policy distribution over legal actions
    probs = self.opponent_policy(state)
    legal_probs = np.array([probs[a] for a in legal], dtype=np.float64)
    legal_probs = np.maximum(legal_probs, 0.0)  # ensure non-negative
    total = legal_probs.sum()
    if total <= 0:
      return self.rng.choice(legal)
    legal_probs /= total  # renormalize
    return self.rng.choices(legal, weights=legal_probs.tolist())[0]

  def _ai_obs(self, state: openspiel.State) -> str:
    return state.observation_string(self.ai_id)

  def reset(self) -> None:
    """Clear history and particles for a new game."""
    self._history.clear()
    self._particles.clear()

  def step(
    self, actor: int, action: int, real_state_after: openspiel.State
  ) -> None:
    """Update belief after a move.

    Args:
        actor: Player who made the move.
        action: Action taken (ignored if actor is opponent).
        real_state_after: State after action for observation extraction.
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

    updated: list[openspiel.State] = []
    for p in self._particles:
      p2 = p.clone()

      if rec.actor_is_ai:
        if rec.ai_action not in p2.legal_actions():
          continue
        p2.apply_action(rec.ai_action)
        if self._ai_obs(p2) == rec.ai_obs_after:
          updated.append(p2)
        continue

      # Opponent action hidden: sample until observation matches
      ok = False
      for _ in range(self.opp_tries_per_particle):
        p3 = p2.clone()
        la = p3.legal_actions()
        if not la:
          break
        a = self._sample_opponent_action(p3)
        p3.apply_action(a)
        if self._ai_obs(p3) == rec.ai_obs_after:
          updated.append(p3)
          ok = True
          break
      if not ok:
        pass

    self._particles = updated
    if not self._particles:
      self._rebuild_particles()

  def sample(self) -> openspiel.State | None:
    """Sample a particle consistent with observation history."""
    if not self._particles:
      return None

    return self.rng.choice(self._particles).clone()

  def _rebuild_particles(self) -> None:
    """Rejection sampling from initial state consistent with observation history."""
    self._particles = []
    tries = 0

    while (
      len(self._particles) < self.num_particles
      and tries < self.rebuild_max_tries
    ):
      tries += 1
      s = self.game.new_initial_state()
      ok = True

      for rec in self._history:
        if rec.actor_is_ai:
          if rec.ai_action not in s.legal_actions():
            ok = False
            break
          s.apply_action(rec.ai_action)
          if self._ai_obs(s) != rec.ai_obs_after:
            ok = False
            break
        else:
          matched = False
          for _ in range(self.opp_tries_per_particle):
            s2 = s.clone()
            la = s2.legal_actions()
            if not la:
              break
            a = self._sample_opponent_action(s2)
            s2.apply_action(a)
            if self._ai_obs(s2) == rec.ai_obs_after:
              s = s2
              matched = True
              break
          if not matched:
            ok = False
            break

      if ok:
        self._particles.append(s)


class ParticleDeterminizationSampler:
  """Adapter conforming ParticleBeliefSampler to DeterminizationSampler.

  Falls back to cloning current state if no particles available.
  """

  def __init__(self, particle_sampler: ParticleBeliefSampler):
    self.particle_sampler = particle_sampler

  def sample(
    self,
    state: openspiel.State,
    rng: random.Random,  # noqa: ARG002
  ) -> openspiel.State:
    """Sample a determinized state from particles or clone."""
    p = self.particle_sampler.sample()
    return p if p is not None else state.clone()
