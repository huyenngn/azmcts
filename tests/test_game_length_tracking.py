#!/usr/bin/env python3
"""Quick test to verify game length tracking works."""

import pathlib
import sys

# Add src to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import openspiel
from scripts.common import config
from scripts.eval import match

# Create a simple game
game = openspiel.Game("tic_tac_toe")

# Simple search config
search_cfg = config.SearchConfig(
  T=8, S=4, c_puct=1.5, dirichlet_alpha=0.0, dirichlet_weight=0.0
)
sampler_cfg = config.SamplerConfig(150, 100, 30)

# Play one game
r0, r1, game_length = match.play_game(
  game=game,
  kind0="random",
  kind1="random",
  search_cfg=search_cfg,
  sampler_cfg=sampler_cfg,
  seed=42,
  device="cpu",
  model_path=None,
  run_id="test",
  game_idx=0,
)

print(f"✓ play_game works: r0={r0}, r1={r1}, length={game_length}")

# Test Result dataclass
result = match.Result()
match.update_result(result, 1.0, -1.0, game_length)

print(
  f"✓ Result tracking works: wins={result.p0_wins}, win_lengths={result.win_lengths}"
)

print("\n✓ All basic tests passed!")
