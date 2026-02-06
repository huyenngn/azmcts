from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from torch.nn import functional as F

import agents
import nets
import openspiel
from scripts.common import agent_factory, config, io, seeding
from utils import utils


def obs_tensor_side_to_move(state: openspiel.State) -> np.ndarray:
  return np.asarray(
    state.observation_tensor(state.current_player()), dtype=np.float32
  )


class Example:
  def __init__(self, obs: np.ndarray, pi: np.ndarray, z: float):
    self.obs = obs
    self.pi = pi
    self.z = z


def self_play_one_game(
  *,
  game: openspiel.Game,
  net: nets.TinyPolicyValueNet,
  search_cfg: config.SearchConfig,
  sampler_cfg: config.SamplerConfig,
  base_seed: int,
  device: str,
  run_id: str,
  game_idx: int,
) -> tuple[list[Example], float]:
  """Play a single self-play game and return examples + player 0 return."""
  state = game.new_initial_state()

  a0, p0 = agent_factory.make_agent(
    kind="azbsmcts",
    player_id=0,
    game=game,
    search_cfg=search_cfg,
    sampler_cfg=sampler_cfg,
    base_seed=base_seed,
    run_id=run_id,
    purpose="train",
    device=device,
    model_path=None,
    net=net,
    game_idx=game_idx,
  )
  a1, p1 = agent_factory.make_agent(
    kind="azbsmcts",
    player_id=1,
    game=game,
    search_cfg=search_cfg,
    sampler_cfg=sampler_cfg,
    base_seed=base_seed,
    run_id=run_id,
    purpose="train",
    device=device,
    model_path=None,
    net=net,
    game_idx=game_idx,
  )

  if (
    p0 is None
    or p1 is None
    or not isinstance(a0, agents.AZBSMCTSAgent)
    or not isinstance(a1, agents.AZBSMCTSAgent)
  ):
    raise ValueError("AZ-BSMCTS agents required for self-play")

  traj: list[tuple[np.ndarray, np.ndarray, int]] = []

  while not state.is_terminal():
    p = state.current_player()
    obs = obs_tensor_side_to_move(state)

    if p == 0:
      action, pi = a0.select_action_with_pi(state)
    else:
      action, pi = a1.select_action_with_pi(state)

    traj.append((obs, pi.astype(np.float32), p))

    actor = state.current_player()
    state.apply_action(action)

    # update both filters
    p0.step(actor=actor, action=action, real_state_after=state)
    p1.step(actor=actor, action=action, real_state_after=state)

  rets = state.returns()
  examples = [Example(obs=obs, pi=pi, z=float(rets[p])) for obs, pi, p in traj]
  return examples, float(rets[0])


def self_play(
  *,
  game: openspiel.Game,
  net: nets.TinyPolicyValueNet,
  num_games: int,
  search_cfg: config.SearchConfig,
  sampler_cfg: config.SamplerConfig,
  base_seed: int,
  device: str,
  run_id: str,
) -> tuple[list[Example], list[float]]:
  """Play multiple self-play games (legacy interface for tune.py)."""
  examples: list[Example] = []
  p0_returns: list[float] = []

  for gi in range(num_games):
    game_examples, p0_ret = self_play_one_game(
      game=game,
      net=net,
      search_cfg=search_cfg,
      sampler_cfg=sampler_cfg,
      base_seed=base_seed,
      device=device,
      run_id=run_id,
      game_idx=gi,
    )
    examples.extend(game_examples)
    p0_returns.append(p0_ret)

  return examples, p0_returns


def train_net(
  *,
  net: nets.TinyPolicyValueNet,
  examples: list[Example],
  epochs: int,
  batch_size: int,
  lr: float,
  device: str,
  seed: int,
  metrics_path: pathlib.Path,
  optimizer_state: dict | None = None,
) -> dict:
  rng = np.random.default_rng(seed)
  opt = torch.optim.Adam(net.parameters(), lr=lr)
  if optimizer_state is not None:
    opt.load_state_dict(optimizer_state)
  net.train()

  obs = np.stack([ex.obs for ex in examples]).astype(np.float32)
  pi = np.stack([ex.pi for ex in examples]).astype(np.float32)
  z = np.array([ex.z for ex in examples], dtype=np.float32)

  obs_t = torch.from_numpy(obs).to(device)
  pi_t = torch.from_numpy(pi).to(device)
  z_t = torch.from_numpy(z).to(device)

  n = obs.shape[0]
  metrics_path.unlink(missing_ok=True)

  for ep in range(epochs):
    idx = rng.permutation(n)
    total_loss = total_ploss = total_vloss = 0.0
    batches = 0

    for start in range(0, n, batch_size):
      batch_idx = idx[start : start + batch_size]
      x = obs_t[batch_idx]
      target_pi = pi_t[batch_idx]
      target_z = z_t[batch_idx].unsqueeze(1)

      logits, v = net(x)
      logp = F.log_softmax(logits, dim=1)
      policy_loss = -(target_pi * logp).sum(dim=1).mean()
      value_loss = F.mse_loss(v, target_z)
      loss = policy_loss + value_loss

      opt.zero_grad(set_to_none=True)
      loss.backward()
      opt.step()

      total_loss += float(loss.item())
      total_ploss += float(policy_loss.item())
      total_vloss += float(value_loss.item())
      batches += 1

    rec = {
      "epoch": ep + 1,
      "loss": total_loss / batches,
      "policy_loss": total_ploss / batches,
      "value_loss": total_vloss / batches,
    }
    with metrics_path.open("a", encoding="utf-8") as f:
      f.write(json.dumps(rec) + "\n")

    print(
      f"epoch {ep + 1}/{epochs}  loss={rec['loss']:.4f}  "
      f"policy={rec['policy_loss']:.4f}  value={rec['value_loss']:.4f}"
    )

  net.eval()
  return opt.state_dict()


def find_latest_checkpoint(
  checkpoints_dir: pathlib.Path,
) -> tuple[pathlib.Path | None, int]:
  """Find latest checkpoint and extract games count.

  Returns:
      (checkpoint_path, games_played) or (None, 0) if no checkpoints found.
  """
  if not checkpoints_dir.exists():
    return None, 0

  checkpoints = sorted(checkpoints_dir.glob("checkpoint_games_*.pt"))
  if not checkpoints:
    return None, 0

  latest = checkpoints[-1]
  # Parse games from filename: checkpoint_games_00042.pt -> 42
  try:
    games = int(latest.stem.replace("checkpoint_games_", ""))
  except ValueError:
    games = 0

  return latest, games


def load_config_from_file(config_path: pathlib.Path) -> config.TrainConfig:
  """Load TrainConfig from saved config.json."""
  payload = io.read_json(config_path)
  cfg_dict = payload["config"]

  game_cfg = config.GameConfig(**cfg_dict["game"])
  search_cfg = config.SearchConfig(**cfg_dict["search"])
  sampler_cfg = config.SamplerConfig(**cfg_dict["sampler"])
  budget_cfg = config.TrainBudget(**cfg_dict["budget"])

  return config.TrainConfig(
    seed=cfg_dict["seed"],
    device=cfg_dict["device"],
    deterministic_torch=cfg_dict["deterministic_torch"],
    run_name=cfg_dict["run_name"],
    out_model_path=cfg_dict["out_model_path"],
    game=game_cfg,
    search=search_cfg,
    budget=budget_cfg,
    lr=cfg_dict["lr"],
    sampler=sampler_cfg,
  )


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", type=str, default="cpu")
  p.add_argument("--deterministic-torch", action="store_true")

  p.add_argument("--game", type=str, default="phantom_go")
  p.add_argument("--game-params", type=str, default='{"board_size": 9}')

  p.add_argument("--games", type=int, default=50)
  p.add_argument(
    "--checkpoint-interval",
    type=int,
    default=0,
    help="Save checkpoint every N games (0 = only final)",
  )
  p.add_argument("--T", type=int, default=8)
  p.add_argument("--S", type=int, default=4)
  p.add_argument("--c-puct", type=float, default=1.5)
  p.add_argument("--dirichlet-alpha", type=float, default=0.03)
  p.add_argument("--dirichlet-weight", type=float, default=0.25)
  p.add_argument("--epochs", type=int, default=5)
  p.add_argument("--batch", type=int, default=64)
  p.add_argument("--lr", type=float, default=1e-4)

  p.add_argument("--runs-root", type=str, default="runs")
  p.add_argument("--run-name", type=str, default="aztrain")
  p.add_argument("--out-model", type=str, default="")  # optional extra copy

  # Resume functionality
  p.add_argument(
    "--run-dir",
    type=str,
    default="",
    help="Fixed directory for this run (disables timestamps)",
  )
  p.add_argument(
    "--resume",
    action="store_true",
    help="Resume from latest checkpoint in run-dir",
  )
  p.add_argument(
    "--replay-max-examples",
    type=int,
    default=0,
    help="Cap replay buffer size (0 = unlimited)",
  )

  p.add_argument("--num-particles", type=int, default=24)
  p.add_argument("--max-matching-opp-actions", type=int, default=24)
  p.add_argument("--rebuild-tries", type=int, default=200)

  args = p.parse_args()

  # Set global seeds and log determinism mode
  seeding.set_global_seeds(
    args.seed, deterministic_torch=args.deterministic_torch, log=True
  )
  seeding.log_repro_fingerprint(args.device)

  game_cfg = config.GameConfig.from_cli(args.game, args.game_params)
  search_cfg = config.SearchConfig(
    T=args.T,
    S=args.S,
    c_puct=args.c_puct,
    dirichlet_alpha=args.dirichlet_alpha,
    dirichlet_weight=args.dirichlet_weight,
  )
  sampler_cfg = config.SamplerConfig(
    num_particles=args.num_particles,
    max_matching_opp_actions=args.max_matching_opp_actions,
    rebuild_max_tries=args.rebuild_tries,
  )

  # Handle run directory creation
  resuming = False
  loaded_cfg: config.TrainConfig | None = None

  if args.run_dir:
    # User specified fixed run directory
    run_dir = pathlib.Path(args.run_dir)
    utils.ensure_dir(run_dir)

    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    figures_dir = run_dir / "figures"
    utils.ensure_dir(checkpoints_dir)
    utils.ensure_dir(eval_dir)
    utils.ensure_dir(figures_dir)

    run = io.RunPaths(
      run_dir=run_dir,
      config_path=run_dir / "config.json",
      train_metrics_path=run_dir / "train_metrics.jsonl",
      checkpoints_dir=checkpoints_dir,
      model_path=run_dir / "model.pt",
      eval_dir=eval_dir,
      figures_dir=figures_dir,
    )

    # Check if we're resuming
    if args.resume and run.config_path.exists():
      loaded_cfg = load_config_from_file(run.config_path)
      resuming = True
  else:
    # Default timestamped directory
    run = io.make_run_dir(args.runs_root, args.run_name, args.seed)

  run_id = run.run_dir.name

  if resuming and loaded_cfg is not None:
    # Use loaded config for consistency
    cfg = loaded_cfg
    print(f"[resume] Using saved config from {run.config_path}")
  else:
    # Create config from CLI args
    cfg = config.TrainConfig(
      seed=args.seed,
      device=args.device,
      deterministic_torch=args.deterministic_torch,
      run_name=args.run_name,
      out_model_path=args.out_model,
      game=game_cfg,
      search=search_cfg,
      budget=config.TrainBudget(
        games=args.games, epochs=args.epochs, batch=args.batch
      ),
      lr=args.lr,
      sampler=sampler_cfg,
    )

    # Write config with reproducibility fingerprint
    fingerprint = seeding.get_repro_fingerprint(args.device)
    config_payload = {
      "config": config.to_jsonable(cfg),
      "fingerprint": fingerprint.to_dict(),
    }
    io.write_json(run.config_path, config_payload)

  game = openspiel.Game(cfg.game.name, cfg.game.params)
  net = nets.TinyPolicyValueNet(
    obs_size=game.observation_tensor_size(),
    num_actions=game.num_distinct_actions(),
  ).to(cfg.device)

  # Checkpoints directory
  checkpoints_dir = run.run_dir / "checkpoints"
  utils.ensure_dir(checkpoints_dir)

  # Initialize training state
  all_examples: list[Example] = []
  all_p0rets: list[float] = []
  games_played = 0
  checkpoint_idx = 0
  optimizer_state: dict | None = None

  # Try to resume if requested
  if args.resume:
    latest_ckpt, ckpt_games = find_latest_checkpoint(checkpoints_dir)
    replay_path = run.run_dir / "replay.pt"

    if latest_ckpt is not None:
      # Load checkpoint weights
      print(f"[resume] Loading checkpoint: {latest_ckpt}")
      net.load_state_dict(
        torch.load(
          str(latest_ckpt),
          map_location=cfg.device,
          weights_only=True,
        )
      )

      # Load replay buffer if it exists
      if replay_path.exists():
        print(f"[resume] Loading replay buffer: {replay_path}")
        replay_data = torch.load(
          str(replay_path), map_location="cpu", weights_only=False
        )

        games_played = replay_data["games_played"]
        optimizer_state = replay_data.get("optimizer_state")

        # Reconstruct examples from saved data
        saved_examples = replay_data["examples"]
        for ex_data in saved_examples:
          all_examples.append(
            Example(
              obs=ex_data["obs"],
              pi=ex_data["pi"],
              z=ex_data["z"],
            )
          )

        all_p0rets = replay_data["p0rets"]

        print(
          f"[resume] Resumed from games={games_played}, "
          f"examples={len(all_examples)}, "
          f"optimizer_state={'loaded' if optimizer_state else 'none'}"
        )
      else:
        print(
          "[resume] Warning: No replay.pt found, starting from checkpoint weights only"
        )
    else:
      print("[resume] No checkpoints found, starting fresh")

  # Interleaved self-play and training (like real AlphaZero)
  interval = (
    args.checkpoint_interval
    if args.checkpoint_interval > 0
    else cfg.budget.games
  )

  while games_played < cfg.budget.games:
    # Determine how many games to play this iteration
    games_this_iter = min(interval, cfg.budget.games - games_played)

    print(
      f"\n=== Self-play games {games_played + 1}-{games_played + games_this_iter} ==="
    )

    for gi in range(games_this_iter):
      game_examples, p0_ret = self_play_one_game(
        game=game,
        net=net,
        search_cfg=cfg.search,
        sampler_cfg=cfg.sampler,
        base_seed=cfg.seed,
        device=cfg.device,
        run_id=run_id,
        game_idx=games_played + gi,
      )
      all_examples.extend(game_examples)
      all_p0rets.append(p0_ret)

    games_played += games_this_iter
    print(f"total examples: {len(all_examples)}, games: {games_played}")
    print(f"p0 return mean: {float(np.mean(all_p0rets)):.3f}")

    # Truncate replay buffer if needed
    if (
      args.replay_max_examples > 0
      and len(all_examples) > args.replay_max_examples
    ):
      print(
        f"[replay] Truncating buffer from {len(all_examples)} to {args.replay_max_examples}"
      )
      all_examples = all_examples[-args.replay_max_examples :]
      all_p0rets = all_p0rets[-args.replay_max_examples :]

    # Train on accumulated data
    print(f"training on {len(all_examples)} examples...")
    optimizer_state = train_net(
      net=net,
      examples=all_examples,
      epochs=cfg.budget.epochs,
      batch_size=cfg.budget.batch,
      lr=cfg.lr,
      device=cfg.device,
      seed=seeding.derive_seed(
        cfg.seed,
        purpose="train/sgd",
        run_id=run_id,
        extra=str(games_played),
      ),
      metrics_path=run.train_metrics_path,
      optimizer_state=optimizer_state,
    )

    # Save checkpoint after this training iteration
    if args.checkpoint_interval > 0 or games_played == cfg.budget.games:
      ckpt_path = checkpoints_dir / f"checkpoint_games_{games_played:05d}.pt"
      torch.save(net.state_dict(), str(ckpt_path))
      print(f"checkpoint: {ckpt_path}")
      checkpoint_idx += 1

      # Save replay buffer atomically
      replay_path = run.run_dir / "replay.pt"
      replay_tmp_path = run.run_dir / "replay.tmp.pt"

      # Convert examples to serializable format
      examples_data = [
        {"obs": ex.obs, "pi": ex.pi, "z": ex.z} for ex in all_examples
      ]

      replay_data = {
        "games_played": games_played,
        "examples": examples_data,
        "p0rets": all_p0rets,
        "optimizer_state": optimizer_state,
      }

      torch.save(replay_data, str(replay_tmp_path))
      replay_tmp_path.replace(replay_path)
      print(f"replay buffer: {replay_path}")

  # Save final model as canonical checkpoint
  torch.save(net.state_dict(), str(run.model_path))
  print(f"\nfinal model: {run.model_path}")
  print(f"run dir: {run.run_dir}")
  print(f"checkpoints: {checkpoint_idx}")

  if cfg.out_model_path:
    outp = pathlib.Path(cfg.out_model_path)
    utils.ensure_dir(outp.parent)
    torch.save(net.state_dict(), str(outp))
    print(f"also saved: {outp}")


if __name__ == "__main__":
  main()
