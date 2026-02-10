from __future__ import annotations

import dataclasses
import logging
import random
from collections import abc as cabc

import numpy as np

import openspiel
from utils import utils

# Type alias for opponent policy function: state -> action probabilities over full action space.
OpponentPolicy = cabc.Callable[[openspiel.State], np.ndarray]

logger = logging.getLogger(__name__)

INITIAL_STATE_SERIALIZED = "\n"

NUM_PARTICLES_INITIAL_FRACTION = 0.7
MATCHES_PER_PARTICLE_INITIAL_FRACTION = 0.7
REBUILD_THRESHOLD_FRACTION = 0.2


@dataclasses.dataclass(frozen=True)
class _StepRecord:
  """Record of one real move from the AI player's perspective."""

  actor_is_ai: bool
  ai_action: int | None
  ai_obs_after: bytes
  checkpoint_particles: list[str]


class ParticleDeterminizationSampler:
  """Belief sampler using particle filtering with optional opponent policy guidance."""

  def __init__(
    self,
    game: openspiel.Game,
    ai_id: int,
    max_num_particles: int = 150,
    max_matches_per_particle: int = 100,
    checkpoint_interval: int = 5,
    rebuild_tries: int = 50,
    seed: int = 0,
    opponent_policy: OpponentPolicy | None = None,
    temperature: float = 1.0,
  ):
    if max_num_particles <= 0:
      raise ValueError("max_num_particles must be > 0")
    if max_matches_per_particle <= 0:
      raise ValueError("max_matches_per_particle must be > 0")
    if checkpoint_interval <= 0:
      raise ValueError("checkpoint_interval must be > 0")
    if rebuild_tries <= 0:
      raise ValueError("rebuild_tries must be > 0")

    self.game = game
    self.ai_id = ai_id
    self.max_num_particles = max_num_particles
    self.max_matches_per_particle = min(
      max_matches_per_particle, game.num_distinct_actions()
    )
    self.checkpoint_interval = checkpoint_interval
    self.rebuild_tries = rebuild_tries
    self.rng = random.Random(seed)
    self.opponent_policy = opponent_policy
    self.temperature = temperature
    self._history: list[_StepRecord] = []
    self._particles: list[str] = []
    self._last_valid_sample: str = INITIAL_STATE_SERIALIZED

  def sample_unique_particles(self, n: int) -> list[openspiel.State]:
    """Return up to n unique particles as deserialized game states."""
    if not self._particles:
      return []

    return [
      self.game.deserialize_state(p)
      for p in self.rng.sample(self._particles, min(n, len(self._particles)))
    ]

  def reset(self) -> None:
    """Clear history and particles for a new game."""
    self._history.clear()
    self._particles.clear()
    self._last_valid_sample = INITIAL_STATE_SERIALIZED

  def step(
    self, actor: int, action: int, real_state_after: openspiel.State
  ) -> None:
    """Update belief after a real move.

    actor/action describe the real move taken in the environment.
    real_state_after is used ONLY to extract the AI observation (non-cheating).
    """
    checkpoint_particles: list[str] = []
    if (len(self._history) + 1) % self.checkpoint_interval == 0:
      checkpoint_particles = self._particles

    rec = _StepRecord(
      actor_is_ai=(actor == self.ai_id),
      ai_action=(action if actor == self.ai_id else None),
      ai_obs_after=self._ai_obs(real_state_after),
      checkpoint_particles=checkpoint_particles,
    )
    self._history.append(rec)

    if not self._particles:
      self._rebuild_particles()

    if rec.actor_is_ai:
      self._particles = self._resample_particles_for_ai(rec)
    else:
      self._particles = self._resample_particles_for_opponent(rec.ai_obs_after)

  def sample(self) -> str:
    """Sample a particle consistent with current observation history."""
    if not self._history:
      return INITIAL_STATE_SERIALIZED

    if not self._particles:
      self._rebuild_particles()

    if not self._particles:
      logger.warning(
        "Particle sampler belief collapsed with empty particles after rebuild; falling back to last valid sample."
      )
      return self._last_valid_sample

    self._last_valid_sample = self.rng.choice(self._particles)
    return self._last_valid_sample

  def _ai_obs(self, state: openspiel.State) -> bytes:
    return np.asarray(
      state.observation_tensor(self.ai_id), dtype=np.float32
    ).tobytes()

  def _get_opponent_action_weights(
    self, state: openspiel.State, temperature: float = 1.0
  ) -> list[float]:
    """Get action weights for opponent moves."""
    legal = state.legal_actions()
    if not legal:
      return []

    if self.opponent_policy is None:
      return (np.ones(len(legal), dtype=np.float32) / len(legal)).tolist()

    policy_probs = self.opponent_policy(state)
    probs = np.array([policy_probs[a] for a in legal], dtype=np.float32)
    probs = np.maximum(probs, 1e-12)

    return utils.apply_temp(probs, temperature=temperature).tolist()

  def _resample_particles_for_ai(
    self, rec: _StepRecord, particles: list[str] | None = None
  ) -> list[str]:
    assert rec.actor_is_ai and rec.ai_action is not None
    if particles is None:
      particles = self._particles
    updated: list[str] = []
    for p in particles:
      s = self.game.deserialize_state(p)
      if rec.ai_action not in s.legal_actions():
        continue
      s.apply_action(rec.ai_action)
      if self._ai_obs(s) == rec.ai_obs_after:
        updated.append(s.serialize())

    return updated

  def _resample_particles_for_opponent(
    self,
    target_obs: bytes,
    particles: list[str] | None = None,
    matches_per_particle: int | None = None,
    num_particles: int | None = None,
    temperature: float = 1.0,
  ) -> list[str]:
    if particles is None:
      particles = self._particles
    if matches_per_particle is None:
      matches_per_particle = self.game.num_distinct_actions()
    if num_particles is None:
      num_particles = self.max_num_particles

    particle_weights: dict[str, float] = {}
    for particle in particles:
      s = self.game.deserialize_state(particle)
      legal = s.legal_actions()
      if not legal:
        continue

      probs = self._get_opponent_action_weights(s, temperature=temperature)

      untried_indices: set[int] = set(range(len(legal)))
      match_count = 0

      while len(untried_indices) > 0:
        if match_count >= matches_per_particle:
          break

        untried = list(untried_indices)
        w = [probs[i] for i in untried]
        action_idx = self.rng.choices(untried, weights=w, k=1)[0]
        untried_indices.remove(action_idx)

        action = legal[action_idx]
        s2 = self.game.deserialize_state(particle)
        s2.apply_action(action)
        if self._ai_obs(s2) != target_obs:
          continue

        weight = probs[action_idx]
        particle_weights[s2.serialize()] = (
          particle_weights.get(s2.serialize(), 0.0) + weight
        )
        match_count += 1

    if not particle_weights:
      return []

    weights = utils.apply_temp(
      np.array(list(particle_weights.values()), dtype=np.float32),
      temperature=temperature,
    )

    return self.rng.choices(
      population=list(particle_weights.keys()),
      weights=list(weights),
      k=num_particles,
    )

  def _rebuild_particles(self) -> None:
    """Rebuild particles using the stored observation history."""
    if not self._history:
      self._particles = []
      return

    logger.debug(
      f"Rebuilding particles with history of {len(self._history)} steps and {len(self._particles)} existing particles."
    )

    num_particles = int(
      self.max_num_particles * NUM_PARTICLES_INITIAL_FRACTION
    )
    matches_per_particle = int(
      self.game.num_distinct_actions() * MATCHES_PER_PARTICLE_INITIAL_FRACTION
    )
    temperature = self.temperature

    latest_checkpoint_n = (len(self._history) - 1) // self.checkpoint_interval
    latest_checkpoint_idx = (latest_checkpoint_n) * self.checkpoint_interval
    checkpoint_step = 1 + latest_checkpoint_n // self.rebuild_tries

    for attempt in range(self.rebuild_tries):
      checkpoint_idx = (
        latest_checkpoint_idx
        - (attempt * checkpoint_step) * self.checkpoint_interval
      )
      particles = [INITIAL_STATE_SERIALIZED]
      start_idx = 0
      if checkpoint_idx > 0:
        checkpoint_particles = self._history[
          checkpoint_idx - 1
        ].checkpoint_particles
        if checkpoint_particles:
          particles = checkpoint_particles
          start_idx = checkpoint_idx - 1
      logger.debug(
        f"Rebuild attempt {attempt + 1}/{self.rebuild_tries} starting from checkpoint index {checkpoint_idx} with {len(particles)} particles."
      )
      history_scale = (
        1.0 + (len(self._history) - start_idx) / self.game.max_game_length()
      )
      num_particles = min(
        int(num_particles * history_scale),
        self.max_num_particles,
      )
      matches_per_particle = min(
        int(matches_per_particle * history_scale),
        self.max_matches_per_particle,
      )
      temperature = temperature * history_scale

      logger.debug(
        f"Rebuilding with num_particles={num_particles} "
        f"matches_per_particle={matches_per_particle} and temperature={temperature:.2f}"
      )

      ok = True
      for idx in range(start_idx, len(self._history)):
        rec = self._history[idx]
        if not particles:
          ok = False
          break

        if rec.actor_is_ai:
          particles = self._resample_particles_for_ai(rec, particles)
        else:
          particles = self._resample_particles_for_opponent(
            rec.ai_obs_after,
            particles,
            matches_per_particle,
            num_particles=num_particles,
            temperature=temperature,
          )

      if ok and particles:
        self._particles.extend(particles)
        return
