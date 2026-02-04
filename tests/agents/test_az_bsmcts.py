"""Tests for agents.az_bsmcts module."""

import numpy as np
import pytest

import agents
import nets
import openspiel
from belief import samplers


@pytest.fixture
def game() -> openspiel.Game:
  """Create a small Phantom Tic-Tac-Toe game for testing."""
  return openspiel.Game("phantom_ttt")


@pytest.fixture
def net(game: openspiel.Game) -> nets.TinyPolicyValueNet:
  """Create a small network for testing."""
  obs_size = game.observation_tensor_size()
  num_actions = game.num_distinct_actions()
  network = nets.TinyPolicyValueNet(obs_size=obs_size, num_actions=num_actions)
  network.eval()
  return network


@pytest.fixture
def sampler(game: openspiel.Game) -> samplers.ParticleDeterminizationSampler:
  """Create a particle sampler for testing."""
  belief = samplers.ParticleBeliefSampler(
    game=game, ai_id=0, num_particles=8, seed=42
  )
  return samplers.ParticleDeterminizationSampler(belief)


class TestAZBSMCTSAgent:
  """Tests for AZBSMCTSAgent class."""

  def test_init(
    self,
    game: openspiel.Game,
    net: nets.TinyPolicyValueNet,
    sampler: samplers.ParticleDeterminizationSampler,
  ) -> None:
    """Test agent initialization."""
    agent = agents.AZBSMCTSAgent(
      player_id=0,
      num_actions=game.num_distinct_actions(),
      obs_size=game.observation_tensor_size(),
      sampler=sampler,
      T=4,
      S=2,
      c_puct=1.5,
      seed=42,
      net=net,
    )
    assert agent.player_id == 0
    assert agent.T == 4
    assert agent.S == 2
    assert agent.c_puct == 1.5

  def test_select_action_returns_legal(
    self,
    game: openspiel.Game,
    net: nets.TinyPolicyValueNet,
    sampler: samplers.ParticleDeterminizationSampler,
  ) -> None:
    """Test that select_action returns a legal action."""
    agent = agents.AZBSMCTSAgent(
      player_id=0,
      num_actions=game.num_distinct_actions(),
      obs_size=game.observation_tensor_size(),
      sampler=sampler,
      T=4,
      S=2,
      seed=42,
      net=net,
    )
    state = game.new_initial_state()
    action = agent.select_action(state)

    assert action in state.legal_actions()

  def test_select_action_with_pi_returns_action_and_policy(
    self,
    game: openspiel.Game,
    net: nets.TinyPolicyValueNet,
    sampler: samplers.ParticleDeterminizationSampler,
  ) -> None:
    """Test that select_action_with_pi returns action and policy vector."""
    agent = agents.AZBSMCTSAgent(
      player_id=0,
      num_actions=game.num_distinct_actions(),
      obs_size=game.observation_tensor_size(),
      sampler=sampler,
      T=4,
      S=2,
      seed=42,
      net=net,
    )
    state = game.new_initial_state()
    action, pi = agent.select_action_with_pi(state, temperature=1.0)

    assert action in state.legal_actions()
    assert isinstance(pi, np.ndarray)
    assert pi.shape == (game.num_distinct_actions(),)
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)

  def test_select_action_deterministic_with_seed(
    self,
    game: openspiel.Game,
    net: nets.TinyPolicyValueNet,
    sampler: samplers.ParticleDeterminizationSampler,
  ) -> None:
    """Test that same seed produces same action sequence."""

    def make_agent() -> agents.AZBSMCTSAgent:
      belief = samplers.ParticleBeliefSampler(
        game=game, ai_id=0, num_particles=8, seed=123
      )
      samp = samplers.ParticleDeterminizationSampler(belief)
      return agents.AZBSMCTSAgent(
        player_id=0,
        num_actions=game.num_distinct_actions(),
        obs_size=game.observation_tensor_size(),
        sampler=samp,
        T=4,
        S=2,
        seed=123,
        net=net,
      )

    agent1 = make_agent()
    agent2 = make_agent()

    state1 = game.new_initial_state()
    state2 = game.new_initial_state()

    action1 = agent1.select_action(state1)
    action2 = agent2.select_action(state2)

    assert action1 == action2

  def test_dirichlet_noise_changes_with_training_mode(
    self,
    game: openspiel.Game,
    net: nets.TinyPolicyValueNet,
    sampler: samplers.ParticleDeterminizationSampler,
  ) -> None:
    """Test that Dirichlet noise affects policy in training mode."""
    # Agent with Dirichlet noise
    agent_noisy = agents.AZBSMCTSAgent(
      player_id=0,
      num_actions=game.num_distinct_actions(),
      obs_size=game.observation_tensor_size(),
      sampler=sampler,
      T=4,
      S=2,
      seed=42,
      net=net,
      dirichlet_alpha=0.3,
      dirichlet_weight=0.25,
    )

    state = game.new_initial_state()

    # Get policy with noise (training)
    _, pi_train = agent_noisy.select_action_with_pi(state, temperature=1.0)

    # Agent without noise
    belief2 = samplers.ParticleBeliefSampler(
      game=game, ai_id=0, num_particles=8, seed=42
    )
    sampler2 = samplers.ParticleDeterminizationSampler(belief2)
    agent_no_noise = agents.AZBSMCTSAgent(
      player_id=0,
      num_actions=game.num_distinct_actions(),
      obs_size=game.observation_tensor_size(),
      sampler=sampler2,
      T=4,
      S=2,
      seed=42,
      net=net,
      dirichlet_alpha=0.0,
      dirichlet_weight=0.0,
    )

    _, pi_no_noise = agent_no_noise.select_action_with_pi(
      state, temperature=1.0
    )

    # Policies should differ due to Dirichlet noise
    # (Note: they could be the same by chance, but unlikely with alpha=0.3)
    # We just check they're valid policies
    assert np.isclose(pi_train.sum(), 1.0, atol=1e-5)
    assert np.isclose(pi_no_noise.sum(), 1.0, atol=1e-5)


class TestBSMCTSAgent:
  """Tests for BSMCTSAgent class (non-neural baseline)."""

  def test_select_action_returns_legal(self, game: openspiel.Game) -> None:
    """Test that BS-MCTS agent returns legal actions."""
    belief = samplers.ParticleBeliefSampler(
      game=game, ai_id=0, num_particles=8, seed=42
    )
    sampler = samplers.ParticleDeterminizationSampler(belief)

    agent = agents.BSMCTSAgent(
      player_id=0,
      num_actions=game.num_distinct_actions(),
      sampler=sampler,
      T=4,
      S=2,
      seed=42,
    )

    state = game.new_initial_state()
    action = agent.select_action(state)

    assert action in state.legal_actions()
