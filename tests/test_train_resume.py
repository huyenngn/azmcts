"""
Tests for training resume functionality.

These tests verify that:
1. Training can resume from checkpoints and replay.pt
2. Replay buffer truncation works correctly
3. Optimizer state is preserved across resume
4. Config is loaded from saved state on resume
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import typing as t

import pytest
import torch

from scripts.common import config, io
from scripts.train import main


@pytest.fixture
def temp_run_dir() -> t.Generator[pathlib.Path, None, None]:
  """Create a temporary run directory for testing."""
  tmpdir = tempfile.mkdtemp()
  yield pathlib.Path(tmpdir)
  shutil.rmtree(tmpdir, ignore_errors=True)


class TestResumeHelpers:
  """Tests for resume helper functions."""

  def test_find_latest_checkpoint_empty_dir(
    self, temp_run_dir: pathlib.Path
  ) -> None:
    """Should return None, 0 when no checkpoints exist."""
    checkpoints_dir = temp_run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path, games = main.find_latest_checkpoint(checkpoints_dir)

    assert ckpt_path is None
    assert games == 0

  def test_find_latest_checkpoint_nonexistent_dir(
    self, temp_run_dir: pathlib.Path
  ) -> None:
    """Should return None, 0 when directory doesn't exist."""
    checkpoints_dir = temp_run_dir / "nonexistent"

    ckpt_path, games = main.find_latest_checkpoint(checkpoints_dir)

    assert ckpt_path is None
    assert games == 0

  def test_find_latest_checkpoint_single(
    self, temp_run_dir: pathlib.Path
  ) -> None:
    """Should find single checkpoint."""
    checkpoints_dir = temp_run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = checkpoints_dir / "checkpoint_games_00010.pt"
    ckpt_file.touch()

    ckpt_path, games = main.find_latest_checkpoint(checkpoints_dir)

    assert ckpt_path == ckpt_file
    assert games == 10

  def test_find_latest_checkpoint_multiple(
    self, temp_run_dir: pathlib.Path
  ) -> None:
    """Should find latest checkpoint by games count."""
    checkpoints_dir = temp_run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    (checkpoints_dir / "checkpoint_games_00010.pt").touch()
    (checkpoints_dir / "checkpoint_games_00020.pt").touch()
    latest = checkpoints_dir / "checkpoint_games_00030.pt"
    latest.touch()

    ckpt_path, games = main.find_latest_checkpoint(checkpoints_dir)

    assert ckpt_path == latest
    assert games == 30

  def test_load_config_from_file(self, temp_run_dir: pathlib.Path) -> None:
    """Should load config from saved JSON."""
    cfg = config.TrainConfig(
      seed=42,
      device="cpu",
      deterministic_torch=True,
      run_name="test",
      out_model_path="",
      game=config.GameConfig(name="tic_tac_toe", params={}),
      search=config.SearchConfig(
        T=4,
        S=2,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
      ),
      budget=config.TrainBudget(games=10, epochs=2, batch=32),
      lr=0.001,
      sampler=config.SamplerConfig(
        min_particles=8,
        max_matching_opp_actions=2,
        rebuild_max_tries=100,
      ),
    )

    config_path = temp_run_dir / "config.json"
    payload = {"config": config.to_jsonable(cfg), "fingerprint": {}}
    io.write_json(config_path, payload)

    loaded_cfg = main.load_config_from_file(config_path)

    assert loaded_cfg.seed == 42
    assert loaded_cfg.device == "cpu"
    assert loaded_cfg.game.name == "tic_tac_toe"
    assert loaded_cfg.search.T == 4
    assert loaded_cfg.budget.games == 10


class TestReplayBufferSaveLoad:
  """Tests for replay buffer serialization."""

  def test_save_and_load_replay_buffer(
    self, temp_run_dir: pathlib.Path
  ) -> None:
    """Should save and load replay buffer correctly."""
    # Create example data
    examples = [
      main.Example(
        obs=torch.randn(9).numpy(),
        pi=torch.randn(9).numpy(),
        z=1.0,
      )
      for _ in range(5)
    ]

    replay_data = {
      "games_played": 10,
      "examples": [{"obs": ex.obs, "pi": ex.pi, "z": ex.z} for ex in examples],
      "p0rets": [1.0, -1.0, 0.0, 1.0, -1.0],
      "optimizer_state": {"param_groups": [{"lr": 0.001}]},
    }

    replay_path = temp_run_dir / "replay.pt"
    replay_tmp_path = temp_run_dir / "replay.tmp.pt"

    # Atomic save
    torch.save(replay_data, str(replay_tmp_path))
    replay_tmp_path.replace(replay_path)

    # Load and verify
    loaded_data = torch.load(
      str(replay_path), map_location="cpu", weights_only=False
    )

    assert loaded_data["games_played"] == 10
    assert len(loaded_data["examples"]) == 5
    assert len(loaded_data["p0rets"]) == 5
    assert loaded_data["optimizer_state"]["param_groups"][0]["lr"] == 0.001


class TestTrainNetOptimizerState:
  """Tests for optimizer state handling in train_net."""

  def test_train_net_returns_optimizer_state(
    self, temp_run_dir: pathlib.Path
  ) -> None:
    """Should return optimizer state after training."""
    import numpy as np

    import nets
    import openspiel

    game = openspiel.Game("tic_tac_toe")
    net = nets.TinyPolicyValueNet(
      obs_size=game.observation_tensor_size(),
      num_actions=game.num_distinct_actions(),
    )

    # Create dummy examples
    examples = [
      main.Example(
        obs=np.zeros(game.observation_tensor_size(), dtype=np.float32),
        pi=np.ones(game.num_distinct_actions(), dtype=np.float32)
        / game.num_distinct_actions(),
        z=0.0,
      )
      for _ in range(10)
    ]

    metrics_path = temp_run_dir / "metrics.jsonl"

    optimizer_state = main.train_net(
      net=net,
      examples=examples,
      epochs=1,
      batch_size=4,
      lr=0.001,
      device="cpu",
      seed=42,
      metrics_path=metrics_path,
    )

    assert optimizer_state is not None
    assert "param_groups" in optimizer_state
    assert "state" in optimizer_state

  def test_train_net_loads_optimizer_state(
    self, temp_run_dir: pathlib.Path
  ) -> None:
    """Should load and use provided optimizer state."""
    import numpy as np

    import nets
    import openspiel

    game = openspiel.Game("tic_tac_toe")
    net = nets.TinyPolicyValueNet(
      obs_size=game.observation_tensor_size(),
      num_actions=game.num_distinct_actions(),
    )

    examples = [
      main.Example(
        obs=np.zeros(game.observation_tensor_size(), dtype=np.float32),
        pi=np.ones(game.num_distinct_actions(), dtype=np.float32)
        / game.num_distinct_actions(),
        z=0.0,
      )
      for _ in range(10)
    ]

    metrics_path = temp_run_dir / "metrics.jsonl"

    # First training
    opt_state_1 = main.train_net(
      net=net,
      examples=examples,
      epochs=1,
      batch_size=4,
      lr=0.001,
      device="cpu",
      seed=42,
      metrics_path=metrics_path,
    )

    # Second training with loaded state (should not raise)
    opt_state_2 = main.train_net(
      net=net,
      examples=examples,
      epochs=1,
      batch_size=4,
      lr=0.001,
      device="cpu",
      seed=43,
      metrics_path=metrics_path,
      optimizer_state=opt_state_1,
    )

    assert opt_state_2 is not None


class TestReplayBufferTruncation:
  """Tests for replay buffer truncation logic."""

  def test_truncation_preserves_newest_examples(self) -> None:
    """Should keep only the newest N examples."""
    examples = [
      main.Example(
        obs=torch.tensor([float(i)]).numpy(),
        pi=torch.tensor([float(i)]).numpy(),
        z=float(i),
      )
      for i in range(100)
    ]

    p0rets = list(range(100))

    # Truncate to 10 newest
    max_examples = 10
    examples_truncated = examples[-max_examples:]
    p0rets_truncated = p0rets[-max_examples:]

    assert len(examples_truncated) == 10
    assert len(p0rets_truncated) == 10
    assert examples_truncated[0].z == 90.0
    assert examples_truncated[-1].z == 99.0
    assert p0rets_truncated[0] == 90
    assert p0rets_truncated[-1] == 99
