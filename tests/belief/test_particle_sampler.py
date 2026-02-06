"""Tests for belief.samplers.particle module."""

import numpy as np
import pytest

import openspiel
from belief import samplers


@pytest.fixture
def game() -> openspiel.Game:
  """Create a small Phantom Tic-Tac-Toe game for testing."""
  return openspiel.Game("phantom_ttt")


class TestParticleBeliefSampler:
  """Test ParticleBeliefSampler class."""

  def test_init_default(self, game: openspiel.Game) -> None:
    """Test default initialization."""
    sampler = samplers.ParticleDeterminizationSampler(game=game, ai_id=0)
    assert sampler.ai_id == 0
    assert sampler.min_particles == 32
    assert sampler.opponent_policy is None

  def test_init_with_opponent_policy(self, game: openspiel.Game) -> None:
    """Test initialization with opponent policy."""

    def dummy_policy(state: openspiel.State) -> np.ndarray:
      return np.ones(9) / 9

    sampler = samplers.ParticleDeterminizationSampler(
      game=game, ai_id=0, opponent_policy=dummy_policy
    )
    assert sampler.opponent_policy is dummy_policy

  def test_sample_opponent_action_uniform(self, game: openspiel.Game) -> None:
    """Test opponent action sampling with uniform random (no policy)."""
    sampler = samplers.ParticleDeterminizationSampler(
      game=game, ai_id=0, seed=42
    )
    state = game.new_initial_state()

    # Sample many actions and verify they're all legal
    actions = sampler._get_opponent_actions_and_weights(state)
    legal = state.legal_actions()
    for a, w in actions:
      assert a in legal

  def test_sample_opponent_action_with_policy(
    self, game: openspiel.Game
  ) -> None:
    """Test opponent action sampling uses policy distribution."""

    # Policy that strongly prefers action 4 (center)
    def biased_policy(state: openspiel.State) -> np.ndarray:
      probs = np.zeros(9)
      probs[4] = 0.99
      for a in state.legal_actions():
        if a != 4:
          probs[a] = 0.001
      return probs

    sampler = samplers.ParticleDeterminizationSampler(
      game=game, ai_id=0, opponent_policy=biased_policy, seed=42
    )
    state = game.new_initial_state()

    # Sample many actions - should heavily favor action 4
    center_count = 0
    for _ in range(1000):
      actions = sampler._get_opponent_actions_and_weights(state)
      if actions[0][0] == 4:
        center_count += 1

    # With 99% probability, center should be chosen most of the time
    assert center_count > 80, f"Expected >80 center moves, got {center_count}"

  def test_sample_opponent_action_handles_zero_probs(
    self, game: openspiel.Game
  ) -> None:
    """Test that zero probabilities are handled gracefully."""

    def zero_policy(state: openspiel.State) -> np.ndarray:
      return np.zeros(9)

    sampler = samplers.ParticleDeterminizationSampler(
      game=game, ai_id=0, opponent_policy=zero_policy, seed=42
    )
    state = game.new_initial_state()

    # Should fall back to uniform when all probs are zero
    action = sampler._get_opponent_actions_and_weights(state)[0][0]
    assert action in state.legal_actions()

  def test_reset_clears_state(self, game: openspiel.Game) -> None:
    """Test that reset clears history and particles."""
    sampler = samplers.ParticleDeterminizationSampler(
      game=game, ai_id=0, seed=42
    )
    state = game.new_initial_state()

    # Make some moves to build history
    state.apply_action(0)
    sampler.step(actor=0, action=0, real_state_after=state)

    assert len(sampler._history) > 0

    sampler.reset()
    assert len(sampler._history) == 0
    assert len(sampler._particles) == 0

  def test_step_builds_particles(self, game: openspiel.Game) -> None:
    """Test that step triggers particle building."""
    sampler = samplers.ParticleDeterminizationSampler(
      game=game, ai_id=0, min_particles=10, seed=42
    )
    state = game.new_initial_state()

    # AI's first move
    state.apply_action(4)
    sampler.step(actor=0, action=4, real_state_after=state)

    # Should have built some particles
    assert len(sampler._particles) > 0

  def test_sample_returns_state(self, game: openspiel.Game) -> None:
    """Test that sample returns a valid state."""
    sampler = samplers.ParticleDeterminizationSampler(
      game=game, ai_id=0, min_particles=10, seed=42
    )

    state = game.new_initial_state()
    state.apply_action(4)
    sampler.step(actor=0, action=4, real_state_after=state)

    sampled = sampler.sample()

    assert sampled is not None
    assert isinstance(sampled, openspiel.State)

  def test_sample_fallback_when_no_particles(
    self, game: openspiel.Game
  ) -> None:
    """Test fallback to cloning when no particles available."""
    sampler = samplers.ParticleDeterminizationSampler(
      game=game, ai_id=0, min_particles=10, seed=42
    )

    # No history, should return initial state
    sampled = sampler.sample()
    assert sampled is not None
