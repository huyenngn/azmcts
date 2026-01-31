"""Tests for belief.samplers.particle module."""

import numpy as np
import pyspiel
import pytest

from belief import samplers


@pytest.fixture
def game() -> pyspiel.Game:
    """Create a small Phantom Tic-Tac-Toe game for testing."""
    return pyspiel.load_game("phantom_ttt")


class TestParticleBeliefSampler:
    """Test ParticleBeliefSampler class."""

    def test_init_default(self, game: pyspiel.Game) -> None:
        """Test default initialization."""
        sampler = samplers.ParticleBeliefSampler(game=game, ai_id=0)
        assert sampler.ai_id == 0
        assert sampler.num_particles == 32
        assert sampler.opponent_policy is None

    def test_init_with_opponent_policy(self, game: pyspiel.Game) -> None:
        """Test initialization with opponent policy."""

        def dummy_policy(state: pyspiel.State) -> np.ndarray:
            return np.ones(9) / 9

        sampler = samplers.ParticleBeliefSampler(
            game=game, ai_id=0, opponent_policy=dummy_policy
        )
        assert sampler.opponent_policy is dummy_policy

    def test_sample_opponent_action_uniform(self, game: pyspiel.Game) -> None:
        """Test opponent action sampling with uniform random (no policy)."""
        sampler = samplers.ParticleBeliefSampler(game=game, ai_id=0, seed=42)
        state = game.new_initial_state()

        # Sample many actions and verify they're all legal
        actions = [sampler._sample_opponent_action(state) for _ in range(100)]
        legal = state.legal_actions()
        for a in actions:
            assert a in legal

    def test_sample_opponent_action_with_policy(
        self, game: pyspiel.Game
    ) -> None:
        """Test opponent action sampling uses policy distribution."""

        # Policy that strongly prefers action 4 (center)
        def biased_policy(state: pyspiel.State) -> np.ndarray:
            probs = np.zeros(9)
            probs[4] = 0.99
            for a in state.legal_actions():
                if a != 4:
                    probs[a] = 0.001
            return probs

        sampler = samplers.ParticleBeliefSampler(
            game=game, ai_id=0, opponent_policy=biased_policy, seed=42
        )
        state = game.new_initial_state()

        # Sample many actions - should heavily favor action 4
        actions = [sampler._sample_opponent_action(state) for _ in range(100)]
        center_count = sum(1 for a in actions if a == 4)

        # With 99% probability, center should be chosen most of the time
        assert center_count > 80, (
            f"Expected >80 center moves, got {center_count}"
        )

    def test_sample_opponent_action_handles_zero_probs(
        self, game: pyspiel.Game
    ) -> None:
        """Test that zero probabilities are handled gracefully."""

        def zero_policy(state: pyspiel.State) -> np.ndarray:
            return np.zeros(9)

        sampler = samplers.ParticleBeliefSampler(
            game=game, ai_id=0, opponent_policy=zero_policy, seed=42
        )
        state = game.new_initial_state()

        # Should fall back to uniform when all probs are zero
        action = sampler._sample_opponent_action(state)
        assert action in state.legal_actions()

    def test_reset_clears_state(self, game: pyspiel.Game) -> None:
        """Test that reset clears history and particles."""
        sampler = samplers.ParticleBeliefSampler(game=game, ai_id=0, seed=42)
        state = game.new_initial_state()

        # Make some moves to build history
        state.apply_action(0)
        sampler.step(actor=0, action=0, real_state_after=state)

        assert len(sampler._history) > 0

        sampler.reset()
        assert len(sampler._history) == 0
        assert len(sampler._particles) == 0

    def test_step_builds_particles(self, game: pyspiel.Game) -> None:
        """Test that step triggers particle building."""
        sampler = samplers.ParticleBeliefSampler(
            game=game, ai_id=0, num_particles=10, seed=42
        )
        state = game.new_initial_state()

        # AI's first move
        state.apply_action(4)
        sampler.step(actor=0, action=4, real_state_after=state)

        # Should have built some particles
        assert len(sampler._particles) > 0


class TestParticleDeterminizationSampler:
    """Test ParticleDeterminizationSampler adapter."""

    def test_sample_returns_state(self, game: pyspiel.Game) -> None:
        """Test that sample returns a valid state."""
        import random

        belief = samplers.ParticleBeliefSampler(
            game=game, ai_id=0, num_particles=10, seed=42
        )
        sampler = samplers.ParticleDeterminizationSampler(belief)

        state = game.new_initial_state()
        state.apply_action(4)
        belief.step(actor=0, action=4, real_state_after=state)

        rng = random.Random(123)
        sampled = sampler.sample(state, rng)

        assert sampled is not None
        assert isinstance(sampled, type(state))

    def test_sample_fallback_when_no_particles(
        self, game: pyspiel.Game
    ) -> None:
        """Test fallback to cloning when no particles available."""
        import random

        belief = samplers.ParticleBeliefSampler(
            game=game, ai_id=0, num_particles=10, rebuild_max_tries=0, seed=42
        )
        sampler = samplers.ParticleDeterminizationSampler(belief)

        state = game.new_initial_state()
        rng = random.Random(123)

        # No particles built yet, should fall back to clone
        sampled = sampler.sample(state, rng)
        assert sampled is not None
