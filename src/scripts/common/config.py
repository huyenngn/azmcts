from __future__ import annotations

import dataclasses
import json
import typing as t


@dataclasses.dataclass(frozen=True)
class GameConfig:
  name: str
  params: dict[str, t.Any]

  @staticmethod
  def from_cli(game: str, game_params_json: str) -> GameConfig:
    params = {}
    if game_params_json:
      params = json.loads(game_params_json)
      if not isinstance(params, dict):
        raise ValueError("--game-params must be a JSON object")
    return GameConfig(name=game, params=params)


@dataclasses.dataclass(frozen=True)
class SearchConfig:
  T: int
  S: int
  c_puct: float
  dirichlet_alpha: float
  dirichlet_weight: float


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
  num_particles: int
  max_matching_opp_actions: int
  rebuild_max_tries: int


@dataclasses.dataclass(frozen=True)
class TrainBudget:
  games: int
  epochs: int
  batch: int


@dataclasses.dataclass(frozen=True)
class TrainConfig:
  seed: int
  device: str
  deterministic_torch: bool
  run_name: str
  out_model_path: str
  game: GameConfig
  search: SearchConfig
  budget: TrainBudget
  lr: float
  sampler: SamplerConfig


@dataclasses.dataclass(frozen=True)
class EvalConfig:
  seed: int
  device: str
  game: GameConfig
  search: SearchConfig
  sampler: SamplerConfig
  n_games: int
  a: str
  b: str
  flip_colors: bool
  model_path: str | None


def to_jsonable(obj: t.Any) -> t.Any:
  if hasattr(obj, "__dataclass_fields__"):
    return dataclasses.asdict(obj)
  return obj
