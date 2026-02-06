"""
Tests for reproducibility guarantees.

These tests verify that:
1. Self-play with the same seed produces the same action sequences.
2. Eval matches with the same seed produce the same outcomes.
3. derive_seed produces consistent outputs.
"""

from __future__ import annotations

import pytest

import agents
import openspiel
from belief import samplers
from scripts.common import seeding


class TestDeriveSeed:
  """Tests for the derive_seed function."""

  def test_deterministic_same_inputs(self) -> None:
    """Same inputs should produce the same seed."""
    seed1 = seeding.derive_seed(
      42, purpose="train/agent", game_idx=0, player_id=0
    )
    seed2 = seeding.derive_seed(
      42, purpose="train/agent", game_idx=0, player_id=0
    )
    assert seed1 == seed2

  def test_different_purpose_different_seed(self) -> None:
    """Different purpose should produce different seeds."""
    seed1 = seeding.derive_seed(
      42, purpose="train/agent", game_idx=0, player_id=0
    )
    seed2 = seeding.derive_seed(
      42, purpose="eval/agent", game_idx=0, player_id=0
    )
    assert seed1 != seed2

  def test_different_game_idx_different_seed(self) -> None:
    """Different game_idx should produce different seeds."""
    seed1 = seeding.derive_seed(
      42, purpose="train/agent", game_idx=0, player_id=0
    )
    seed2 = seeding.derive_seed(
      42, purpose="train/agent", game_idx=1, player_id=0
    )
    assert seed1 != seed2

  def test_different_player_id_different_seed(self) -> None:
    """Different player_id should produce different seeds."""
    seed1 = seeding.derive_seed(
      42, purpose="train/agent", game_idx=0, player_id=0
    )
    seed2 = seeding.derive_seed(
      42, purpose="train/agent", game_idx=0, player_id=1
    )
    assert seed1 != seed2

  def test_different_base_seed_different_seed(self) -> None:
    """Different base seed should produce different seeds."""
    seed1 = seeding.derive_seed(
      42, purpose="train/agent", game_idx=0, player_id=0
    )
    seed2 = seeding.derive_seed(
      43, purpose="train/agent", game_idx=0, player_id=0
    )
    assert seed1 != seed2

  def test_seed_is_32bit(self) -> None:
    """Derived seed should be a 32-bit unsigned integer."""
    seed = seeding.derive_seed(42, purpose="test")
    assert 0 <= seed < 2**32


class TestBSMCTSReproducibility:
  """Tests for BS-MCTS agent reproducibility."""

  @pytest.fixture
  def game(self) -> openspiel.Game:
    """Create a small phantom game for testing."""
    # Use phantom_ttt for faster tests
    return openspiel.Game("phantom_ttt")

  def _create_agent_with_seed(
    self, game: openspiel.Game, player_id: int, seed: int
  ) -> tuple[agents.BSMCTSAgent, samplers.ParticleDeterminizationSampler]:
    """Create a BS-MCTS agent with deterministic seeding."""
    particle_seed = seeding.derive_seed(
      seed, purpose="test/belief", game_idx=0, player_id=player_id
    )
    sampler = samplers.ParticleDeterminizationSampler(
      game=game,
      ai_id=player_id,
      min_particles=8,
      rebuild_max_tries=50,
      seed=particle_seed,
    )
    agent_seed = seeding.derive_seed(
      seed, purpose="test/agent", game_idx=0, player_id=player_id
    )
    agent = agents.BSMCTSAgent(
      player_id=player_id,
      num_actions=game.num_distinct_actions(),
      sampler=sampler,
      T=4,  # Small T for fast tests
      S=2,
      seed=agent_seed,
    )
    return agent, sampler

  def _play_game(
    self, game: openspiel.Game, seed: int
  ) -> tuple[list[int], list[float]]:
    """Play a game and return action sequence and returns."""
    a0, p0 = self._create_agent_with_seed(game, 0, seed)
    a1, p1 = self._create_agent_with_seed(game, 1, seed)

    state = game.new_initial_state()
    actions: list[int] = []

    while not state.is_terminal():
      actor = state.current_player()
      if actor == 0:
        action = a0.select_action(state)
      else:
        action = a1.select_action(state)

      actions.append(action)
      state.apply_action(action)

      p0.step(actor=actor, action=action, real_state_after=state)
      p1.step(actor=actor, action=action, real_state_after=state)

    return actions, list(state.returns())

  def test_same_seed_same_actions(self, game: openspiel.Game) -> None:
    """Playing with the same seed should produce the same action sequence."""
    seed = 12345
    actions1, returns1 = self._play_game(game, seed)
    actions2, returns2 = self._play_game(game, seed)

    assert actions1 == actions2, "Action sequences should be identical"
    assert returns1 == returns2, "Returns should be identical"

  def test_different_seed_likely_different_actions(
    self, game: openspiel.Game
  ) -> None:
    """Playing with different seeds should likely produce different actions."""
    actions1, _ = self._play_game(game, 12345)
    actions2, _ = self._play_game(game, 54321)

    # Note: There's a small chance they could be the same by coincidence,
    # but with MCTS it's very unlikely
    assert actions1 != actions2, (
      "Action sequences should differ with different seeds"
    )


class TestReproFingerprint:
  """Tests for reproducibility fingerprint."""

  def test_fingerprint_has_required_fields(self) -> None:
    """Fingerprint should contain all required fields."""

    fp = seeding.get_repro_fingerprint("cpu")
    d = fp.to_dict()

    assert "python_version" in d
    assert "platform" in d
    assert "torch_version" in d
    assert "git_commit" in d

  def test_fingerprint_to_json(self) -> None:
    """Fingerprint should serialize to valid JSON."""
    import json

    fp = seeding.get_repro_fingerprint("cpu")
    json_str = fp.to_json()

    # Should not raise
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)


class TestSetGlobalSeeds:
  """Tests for set_global_seeds function."""

  def test_set_global_seeds_determinism(self) -> None:
    """Global seeds should produce deterministic numpy random."""
    import numpy as np

    seeding.set_global_seeds(42, deterministic_torch=False, log=False)
    vals1 = np.random.rand(5).tolist()

    seeding.set_global_seeds(42, deterministic_torch=False, log=False)
    vals2 = np.random.rand(5).tolist()

    assert vals1 == vals2

  def test_set_global_seeds_torch_determinism(self) -> None:
    """Global seeds should produce deterministic torch random."""
    import torch

    seeding.set_global_seeds(42, deterministic_torch=False, log=False)
    vals1 = torch.rand(5).tolist()

    seeding.set_global_seeds(42, deterministic_torch=False, log=False)
    vals2 = torch.rand(5).tolist()

    assert vals1 == vals2
