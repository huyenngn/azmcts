from __future__ import annotations

import argparse
import datetime
import pathlib

import pyspiel

from scripts.common import config, io, seeding
from scripts.eval import match


def az_winrate(res: dict, az_label: str = "azbsmcts") -> float:
    items = list(res.items())
    r1 = items[0][1]
    r2 = items[1][1]
    # r1/r2 are Result dataclasses in-process; we only need wins and games
    az_wins = (
        (r1.p0_wins + r2.p1_wins)
        if az_label in items[0][0]
        else (r1.p1_wins + r2.p0_wins)
    )
    total = r1.games + r2.games
    return az_wins / max(1, total)


def find_checkpoints(run_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find all checkpoint files in a run directory."""
    checkpoints = []
    # Main model
    if (run_dir / "model.pt").exists():
        checkpoints.append(run_dir / "model.pt")
    # Intermediate checkpoints
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        checkpoints.extend(sorted(ckpt_dir.glob("checkpoint_games_*.pt")))
    return checkpoints


def extract_games_from_checkpoint(path: pathlib.Path) -> int | None:
    """Extract game count from checkpoint filename."""
    name = path.stem
    if name == "model":
        return None  # Will use config.json
    if name.startswith("checkpoint_games_"):
        try:
            return int(name.replace("checkpoint_games_", ""))
        except ValueError:
            return None
    return None


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

    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a single training run directory (e.g., runs/run_20260130_123456)",
    )

    p.add_argument("--x-axis", type=str, default="games")  # from run config
    p.add_argument("--num-particles", type=int, default=32)
    p.add_argument("--opp-tries", type=int, default=8)
    p.add_argument("--rebuild-tries", type=int, default=200)

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
        args.num_particles, args.opp_tries, args.rebuild_tries
    )

    game = pyspiel.load_game(game_cfg.name, game_cfg.params)

    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return
    if not (run_dir / "model.pt").exists():
        print(f"No model.pt found in {run_dir}")
        return

    cfg_path = run_dir / "config.json"
    cfg = io.read_json(cfg_path) if cfg_path.exists() else {}

    # Get total games from config for final model
    total_games = float("nan")
    if isinstance(cfg, dict):
        if "config" in cfg and isinstance(cfg["config"], dict):
            inner = cfg["config"]
            if "games" in inner:
                total_games = float(inner["games"])
            elif (
                "budget" in inner
                and isinstance(inner["budget"], dict)
                and "games" in inner["budget"]
            ):
                total_games = float(inner["budget"]["games"])
        elif "games" in cfg:
            total_games = float(cfg["games"])
        elif (
            "budget" in cfg
            and isinstance(cfg["budget"], dict)
            and "games" in cfg["budget"]
        ):
            total_games = float(cfg["budget"]["games"])

    # Find all checkpoints (intermediate + final)
    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return

    rows = []
    sweep_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for ckpt_path in checkpoints:
        # Determine games count for this checkpoint
        games_from_name = extract_games_from_checkpoint(ckpt_path)
        x_games = games_from_name if games_from_name else total_games

        model_path = str(ckpt_path)
        ckpt_id = ckpt_path.name

        # Evaluate vs BS-MCTS
        res_b = match.run_match(
            game=game,
            a="azbsmcts",
            b="bsmcts",
            n=args.n,
            search_cfg=search_cfg,
            sampler_cfg=sampler_cfg,
            seed=args.seed,
            device=args.device,
            model_path=model_path,
            run_id=f"sweep_{sweep_id}_{ckpt_id}",
        )
        wr_b = az_winrate(res_b)

        # Evaluate vs random
        random_seed = seeding.derive_seed(
            args.seed, purpose="eval/sweep_random", extra=ckpt_id
        )
        res_r = match.run_match(
            game=game,
            a="azbsmcts",
            b="random",
            n=args.n,
            search_cfg=search_cfg,
            sampler_cfg=sampler_cfg,
            seed=random_seed,
            device=args.device,
            model_path=model_path,
            run_id=f"sweep_{sweep_id}_{ckpt_id}_random",
        )
        wr_r = az_winrate(res_r)

        row = {
            "checkpoint": ckpt_path.name,
            "model_path": model_path,
            "x_games": float(x_games) if x_games else float("nan"),
            "wr_vs_bsmcts": float(wr_b),
            "wr_vs_random": float(wr_r),
        }
        rows.append(row)
        print(
            f"{ckpt_id}: games={x_games} "
            f"wr_vs_bsmcts={wr_b:.3f} wr_vs_random={wr_r:.3f}"
        )

    out_path = run_dir / "eval_sweep.jsonl"
    io.write_jsonl(out_path, rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
