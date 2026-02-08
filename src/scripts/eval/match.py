# scripts/eval/match.py
from __future__ import annotations

import argparse
import dataclasses
import datetime
import pathlib
import random

import openspiel
from scripts.common import agent_factory, config, io, seeding
from utils import utils


@dataclasses.dataclass
class Result:
  games: int = 0
  p0_wins: int = 0
  p1_wins: int = 0
  draws: int = 0
  p0_return_sum: float = 0.0
  p1_return_sum: float = 0.0
  win_lengths: list[int] = dataclasses.field(default_factory=list)
  loss_lengths: list[int] = dataclasses.field(default_factory=list)
  draw_lengths: list[int] = dataclasses.field(default_factory=list)


def update_result(res: Result, r0: float, r1: float, game_length: int) -> None:
  res.games += 1
  res.p0_return_sum += r0
  res.p1_return_sum += r1
  if r0 > r1:
    res.p0_wins += 1
    res.win_lengths.append(game_length)
  elif r1 > r0:
    res.p1_wins += 1
    res.loss_lengths.append(game_length)
  else:
    res.draws += 1
    res.draw_lengths.append(game_length)


def play_game(
  *,
  game: openspiel.Game,
  kind0: str,
  kind1: str,
  search_cfg: config.SearchConfig,
  sampler_cfg: config.SamplerConfig,
  seed: int,
  device: str,
  model_path: str | None,
  run_id: str,
  game_idx: int,
) -> tuple[float, float, int]:
  rng = random.Random(
    seeding.derive_seed(
      seed, purpose="eval/rng", run_id=run_id, game_idx=game_idx
    )
  )

  state = game.new_initial_state()
  a0, p0 = agent_factory.make_agent(
    kind=kind0,
    player_id=0,
    game=game,
    search_cfg=search_cfg,
    sampler_cfg=sampler_cfg,
    base_seed=seed,
    run_id=run_id,
    purpose="eval",
    device=device,
    model_path=model_path,
    game_idx=game_idx,
  )
  a1, p1 = agent_factory.make_agent(
    kind=kind1,
    player_id=1,
    game=game,
    search_cfg=search_cfg,
    sampler_cfg=sampler_cfg,
    base_seed=seed,
    run_id=run_id,
    purpose="eval",
    device=device,
    model_path=model_path,
    game_idx=game_idx,
  )

  while not state.is_terminal():
    actor = state.current_player()
    action = agent_factory.select_action(
      kind0 if actor == 0 else kind1,
      a0 if actor == 0 else a1,
      state,
      rng,
    )
    state.apply_action(action)

    if p0 is not None:
      p0.step(actor=actor, action=action, real_state_after=state)
    if p1 is not None:
      p1.step(actor=actor, action=action, real_state_after=state)

  r = state.returns()
  game_length = state.game_length()
  return float(r[0]), float(r[1]), game_length


def run_match(
  *,
  game: openspiel.Game,
  a: str,
  b: str,
  n: int,
  search_cfg: config.SearchConfig,
  sampler_cfg: config.SamplerConfig,
  seed: int,
  device: str,
  model_path: str | None,
  run_id: str,
) -> dict[str, Result]:
  out: dict[str, Result] = {}

  res_ab = Result()
  for i in range(n):
    r0, r1, game_length = play_game(
      game=game,
      kind0=a,
      kind1=b,
      search_cfg=search_cfg,
      sampler_cfg=sampler_cfg,
      seed=seed,
      device=device,
      model_path=model_path,
      run_id=run_id,
      game_idx=i,
    )
    update_result(res_ab, r0, r1, game_length)
  out[f"{a}(p0) vs {b}(p1)"] = res_ab

  res_ba = Result()
  for i in range(n):
    r0, r1, game_length = play_game(
      game=game,
      kind0=b,
      kind1=a,
      search_cfg=search_cfg,
      sampler_cfg=sampler_cfg,
      seed=seed,
      device=device,
      model_path=model_path,
      run_id=run_id,
      game_idx=10_000 + i,
    )
    update_result(res_ba, r0, r1, game_length)
  out[f"{b}(p0) vs {a}(p1)"] = res_ba

  return out


def summarize(label: str, r: Result) -> None:
  p0_wr = r.p0_wins / max(1, r.games)
  p1_wr = r.p1_wins / max(1, r.games)
  dr = r.draws / max(1, r.games)
  print(f"\n{label}")
  print(f"games: {r.games}")
  print(f"p0 winrate: {p0_wr:.3f} | p1 winrate: {p1_wr:.3f} | draws: {dr:.3f}")
  print(
    f"mean returns: p0={r.p0_return_sum / r.games:.3f}, p1={r.p1_return_sum / r.games:.3f}"
  )


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", type=str, default="cpu")

  p.add_argument("--game", type=str, default="phantom_go")
  p.add_argument("--game-params", type=str, default='{"board_size": 9}')

  p.add_argument("--n", type=int, default=20)
  p.add_argument("--T", type=int, default=8)
  p.add_argument("--S", type=int, default=4)
  p.add_argument("--c-puct", type=float, default=1.5)
  p.add_argument("--dirichlet-alpha", type=float, default=0.0)
  p.add_argument("--dirichlet-weight", type=float, default=0.0)

  p.add_argument("--a", type=str, default="azbsmcts")
  p.add_argument("--b", type=str, default="bsmcts")
  p.add_argument("--flip-colors", action="store_true")

  p.add_argument("--model", type=str, default=None)
  p.add_argument("--out-json", type=str, default="")

  p.add_argument("--max-num-particles", type=int, default=150)
  p.add_argument("--max-matches-per-particle", type=int, default=100)
  p.add_argument("--rebuild-tries", type=int, default=30)

  args = p.parse_args()

  game_cfg = config.GameConfig.from_cli(args.game, args.game_params)
  search_cfg = config.SearchConfig(
    T=args.T,
    S=args.S,
    c_puct=args.c_puct,
    dirichlet_alpha=args.dirichlet_alpha,
    dirichlet_weight=args.dirichlet_weight,
  )
  sampler_cfg = config.SamplerConfig(
    args.max_num_particles, args.max_matches_per_particle, args.rebuild_tries
  )

  game = openspiel.Game(game_cfg.name, game_cfg.params)

  run_id = f"match_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
  cfg = config.EvalConfig(
    seed=args.seed,
    device=args.device,
    game=game_cfg,
    search=search_cfg,
    sampler=sampler_cfg,
    n_games=args.n,
    a=args.a,
    b=args.b,
    flip_colors=args.flip_colors,
    model_path=args.model,
  )

  # Include reproducibility fingerprint in output
  fingerprint = seeding.get_repro_fingerprint(args.device)
  payload = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "config": config.to_jsonable(cfg),
    "fingerprint": fingerprint.to_dict(),
    "results": [],
  }

  matchups = [(args.a, args.b)]
  if args.flip_colors:
    matchups.append((args.b, args.a))

  for a, b in matchups:
    res = run_match(
      game=game,
      a=a,
      b=b,
      n=args.n,
      search_cfg=search_cfg,
      sampler_cfg=sampler_cfg,
      seed=args.seed,
      device=args.device,
      model_path=args.model,
      run_id=run_id,
    )
    for label, r in res.items():
      summarize(label, r)
      payload["results"].append(
        {
          "label": label,
          "games": r.games,
          "p0_wins": r.p0_wins,
          "p1_wins": r.p1_wins,
          "draws": r.draws,
          "p0_mean_return": r.p0_return_sum / r.games,
          "p1_mean_return": r.p1_return_sum / r.games,
        }
      )

  if args.out_json:
    outp = pathlib.Path(args.out_json)
    utils.ensure_dir(outp.parent)
    io.write_json(outp, payload)
    print(f"\nwrote: {outp}")


if __name__ == "__main__":
  main()
