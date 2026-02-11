from __future__ import annotations

import collections
import dataclasses
import logging
import random
from collections import abc as cabc

import numpy as np

import openspiel
from utils import utils

OpponentPolicy = cabc.Callable[[list[openspiel.State]], list[np.ndarray]]

logger = logging.getLogger(__name__)

INITIAL_STATE_SERIALIZED = "\n"

INITIAL_BUDGET_FRACTION = 0.7


@dataclasses.dataclass
class _StepRecord:
  """Record of one real move from the AI player's perspective."""

  actor_is_ai: bool
  ai_action: int | None
  ai_obs_after: bytes
  checkpoint_particles: list[str]
  particle_diversity: float


class ParticleDeterminizationSampler:
  """Belief sampler using particle filtering with optional opponent policy guidance."""

  def __init__(
    self,
    game: openspiel.Game,
    ai_id: int,
    max_num_particles: int = 150,
    max_matches_per_particle: int = 100,
    checkpoint_interval: int = 5,
    rebuild_tries: int = 10,
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
    self._checkpoint_indices: set[int] = set()

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
    self._checkpoint_indices.clear()

  def step(
    self, actor: int, action: int, real_state_after: openspiel.State
  ) -> None:
    """Update belief after a real move.

    actor/action describe the real move taken in the environment.
    real_state_after is used ONLY to extract the AI observation (non-cheating).
    """
    checkpoint_particles: list[str] = []
    if self._particles and (
      len(self._history) % self.checkpoint_interval == 0
    ):
      checkpoint_particles = self._particles
      self._checkpoint_indices.add(len(self._history))

    rec = _StepRecord(
      actor_is_ai=(actor == self.ai_id),
      ai_action=(action if actor == self.ai_id else None),
      ai_obs_after=self._ai_obs(real_state_after),
      checkpoint_particles=checkpoint_particles,
      particle_diversity=self._get_particle_diversity(self._particles),
    )
    self._history.append(rec)

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

  def _get_particle_diversity(
    self, particles: list[str] | None = None
  ) -> float:
    if particles is None:
      particles = self._particles
    if not particles:
      return 0.0
    return len(set(particles)) / len(particles)

  def _get_or_create_transition(
    self, parent: str, action: int
  ) -> tuple[str, bytes] | None:
    """Get or compute (child_state, observation) for a transition.

    Returns None if the action is not legal for this parent state.
    """
    s = self.game.deserialize_state(parent)
    if action not in s.legal_actions():
      return None
    s.apply_action(action)
    obs = self._ai_obs(s)
    child_str = s.serialize()
    return child_str, obs

  def _resample_particles_for_ai(
    self,
    rec: _StepRecord,
    particles: list[str] | None = None,
    num_particles: int | None = None,
  ) -> list[str]:
    assert rec.actor_is_ai and rec.ai_action is not None
    if particles is None:
      particles = self._particles
    if num_particles is None:
      num_particles = len(particles)

    surviving: list[str] = []
    for p in set(particles):
      result = self._get_or_create_transition(p, rec.ai_action)
      if result is None:
        continue
      child_str, obs = result
      if obs == rec.ai_obs_after:
        surviving.append(child_str)

    if surviving:
      return self.rng.choices(surviving, k=num_particles)
    return []

  def _get_opponent_action_weights(
    self,
    states: list[openspiel.State],
  ) -> list[list[float]]:
    """Get action weights for states."""
    if not states:
      return []

    if self.opponent_policy is not None:
      all_policy_probs = self.opponent_policy(states)
      results: list[list[float]] = []
      for state, policy_probs in zip(states, all_policy_probs, strict=True):
        legal = state.legal_actions()
        if not legal:
          results.append([])
          continue
        probs = np.array([policy_probs[a] for a in legal], dtype=np.float32)
        probs = np.maximum(probs, 1e-12)
        results.append(probs.tolist())
      return results

    return [
      (
        np.ones(len(s.legal_actions()), dtype=np.float32)
        / len(s.legal_actions())
      ).tolist()
      for s in states
    ]

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

    particle_counts = collections.Counter(particles)

    particle_data: list[tuple[str, list[int], int]] = []
    deserialized_states: list[openspiel.State] = []
    for particle, count in particle_counts.items():
      s = self.game.deserialize_state(particle)
      legal = s.legal_actions()
      if legal:
        particle_data.append((particle, legal, count))
        deserialized_states.append(s)

    if not particle_data:
      return []

    if self.game.name == "phantom_go":
      tmp_data = particle_data[0]
      tmp = self.game.deserialize_state(tmp_data[0])
      tmp.apply_action(tmp_data[1][-1])
      if target_obs == self._ai_obs(tmp):
        results: list[str] = []
        for s, (_, legal, _count) in zip(
          deserialized_states, particle_data, strict=True
        ):
          s.apply_action(legal[-1])
          results.append(s.serialize())
        return results

    all_probs = self._get_opponent_action_weights(deserialized_states)

    particle_weights: dict[str, float] = {}
    for (particle, legal, count), probs in zip(
      particle_data, all_probs, strict=True
    ):
      if not probs:
        continue

      if self.game.name == "phantom_go":
        untried_indices = set(range(len(legal) - 1))
      else:
        untried_indices = set(range(len(legal)))
      match_count = 0

      while len(untried_indices) > 0:
        if match_count >= matches_per_particle:
          break

        untried = list(untried_indices)
        w = [probs[i] for i in untried]
        action_idx = self.rng.choices(untried, weights=w, k=1)[0]
        untried_indices.remove(action_idx)

        action = legal[action_idx]
        result = self._get_or_create_transition(particle, action)
        if result is None:
          continue
        child_str, obs = result
        if obs != target_obs:
          continue

        # Scale weight by duplicate count so common parents contribute more
        weight = probs[action_idx] * count
        particle_weights[child_str] = (
          particle_weights.get(child_str, 0.0) + weight
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

    sorted_checkpoint_indices = list(self._checkpoint_indices)
    sorted_checkpoint_indices.sort()
    logger.debug(f"checkpoint indices: {sorted_checkpoint_indices}")
    for attempt in range(self.rebuild_tries):
      if (tmp := (1 + attempt)) <= len(
        sorted_checkpoint_indices
      ) and attempt < self.rebuild_tries - 1:
        start_idx = sorted_checkpoint_indices[-tmp]
        particles = self._history[start_idx].checkpoint_particles
      else:
        particles = [INITIAL_STATE_SERIALIZED]
        start_idx = 0

      logger.debug(
        f"Rebuild attempt {attempt + 1}/{self.rebuild_tries} starting from index {start_idx} with {len(particles)} particles."
      )
      history_scale = (
        1.0 + (len(self._history) - start_idx) / self.game.max_game_length()
      )
      attempt_scale = 1.0 + attempt / self.rebuild_tries
      num_particles = min(
        int(self.max_num_particles * INITIAL_BUDGET_FRACTION * history_scale),
        self.max_num_particles,
      )
      matches_per_particle = min(
        int(
          self.max_matches_per_particle
          * INITIAL_BUDGET_FRACTION
          * history_scale
        ),
        self.max_matches_per_particle,
      )
      temperature = self.temperature * history_scale * attempt_scale

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
          particles = self._resample_particles_for_ai(
            rec, particles, num_particles=num_particles
          )
        else:
          particles = self._resample_particles_for_opponent(
            rec.ai_obs_after,
            particles,
            matches_per_particle,
            num_particles=num_particles,
            temperature=temperature,
          )

        if (idx != 0) and (idx % self.checkpoint_interval == 0):
          diversity = self._get_particle_diversity(particles)
          if rec.particle_diversity < diversity:
            logger.debug(
              f"Updating checkpoint at index {idx} with diversity {diversity:.2%}"
            )
            rec.particle_diversity = diversity
            rec.checkpoint_particles = particles
            self._checkpoint_indices.add(idx)

      if ok and particles:
        if attempt > 0:
          checkpoint_idx = len(self._history) - 1
          if checkpoint_idx in self._checkpoint_indices:
            self._checkpoint_indices.remove(checkpoint_idx)
          self._checkpoint_indices.add(checkpoint_idx)
          rec = self._history[checkpoint_idx]
          rec.checkpoint_particles = particles
          rec.particle_diversity = self._get_particle_diversity(particles)
          logger.debug(
            f"Final checkpoint updated with {len(particles)} particles and diversity {rec.particle_diversity:.2%}"
          )

        self._particles = particles
        return
