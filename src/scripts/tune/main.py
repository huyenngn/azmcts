import argparse
import datetime
import json
import pathlib

import numpy as np
import optuna
import pyspiel
import torch

import nets
from scripts import train
from scripts.common import config, seeding
from scripts.eval import match
from utils import utils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--board", type=int, default=9)

    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)

    parser.add_argument("--eval-n", type=int, default=10)

    parser.add_argument("--study-name", type=str, default="az_tune")
    parser.add_argument(
        "--storage", type=str, default="sqlite:///runs/optuna.db"
    )
    parser.add_argument("--direction", type=str, default="maximize")
    args = parser.parse_args()

    utils.ensure_dir(pathlib.Path("runs"))
    seeding.set_global_seeds(args.seed, deterministic_torch=False, log=True)

    game = pyspiel.load_game("phantom_go", {"board_size": args.board})
    num_actions = game.num_distinct_actions()

    # FIXED ARCHITECTURE (keep tuning simple and checkpoints compatible)
    FIXED_HIDDEN = 256

    def objective(trial: optuna.Trial) -> float:
        """Single trial: train briefly and evaluate against BS-MCTS."""
        # MCTS parameters
        T = trial.suggest_int("T", 2, 16, log=True)
        S = trial.suggest_int("S", 2, 8)
        c_puct = trial.suggest_float("c_puct", 0.5, 3.0, log=True)
        dirichlet_alpha = trial.suggest_float(
            "dirichlet_alpha", 0.01, 0.5, log=True
        )

        # Training parameters
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        temp = trial.suggest_float("temp", 0.5, 1.5)

        # Belief sampler parameters
        num_particles = trial.suggest_int("num_particles", 10, 64, log=True)

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = pathlib.Path("runs") / f"{ts}_trial{trial.number:04d}"
        utils.ensure_dir(run_dir)

        cfg = {
            "trial": trial.number,
            "params": dict(trial.params),
            "budget": {
                "games": args.games,
                "epochs": args.epochs,
                "batch": args.batch,
                "eval_n": args.eval_n,
            },
            "seed": args.seed,
            "device": args.device,
            "board": args.board,
            "arch": {"hidden": FIXED_HIDDEN},
        }
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        net = nets.TinyPolicyValueNet(
            obs_size=game.observation_tensor_size(),
            num_actions=num_actions,
            hidden=FIXED_HIDDEN,
        ).to(args.device)

        # Create config objects for this trial
        search_cfg = config.SearchConfig(
            T=T,
            S=S,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_weight=0.25,  # Fixed, only alpha is tuned
        )
        sampler_cfg = config.SamplerConfig(
            num_particles=num_particles,
            opp_tries_per_particle=8,
            rebuild_max_tries=200,
        )
        run_id = f"trial{trial.number:04d}"

        # Derive seeds deterministically for each trial
        selfplay_seed = seeding.derive_seed(
            args.seed, purpose="tune/selfplay", extra=str(trial.number)
        )
        examples, p0rets = train.self_play(
            game=game,
            net=net,
            num_games=args.games,
            search_cfg=search_cfg,
            sampler_cfg=sampler_cfg,
            base_seed=selfplay_seed,
            temperature=temp,
            device=args.device,
            run_id=run_id,
        )

        metrics_path = run_dir / "train_metrics.jsonl"
        sgd_seed = seeding.derive_seed(
            args.seed, purpose="tune/sgd", extra=str(trial.number)
        )
        train.train_net(
            net=net,
            examples=examples,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=lr,
            device=args.device,
            seed=sgd_seed,
            metrics_path=metrics_path,
        )

        model_path = run_dir / "model.pt"
        torch.save(net.state_dict(), str(model_path))

        # Evaluate vs baseline with deterministically derived seed
        eval_seed = seeding.derive_seed(
            args.seed, purpose="tune/eval", extra=str(trial.number)
        )
        res = match.run_match(
            game=game,
            a="azbsmcts",
            b="bsmcts",
            n=args.eval_n,
            search_cfg=search_cfg,
            sampler_cfg=sampler_cfg,
            seed=eval_seed,
            device=args.device,
            model_path=str(model_path),
            run_id=run_id,
        )

        items = list(res.items())
        r1 = items[0][1]  # az as p0
        r2 = items[1][1]  # az as p1
        az_wins = r1.p0_wins + r2.p1_wins
        total = r1.games + r2.games
        winrate = az_wins / max(1, total)

        with (run_dir / "eval.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "winrate_az_vs_bsmcts": float(winrate),
                    "p0_return_mean_selfplay": float(np.mean(p0rets))
                    if p0rets
                    else None,
                    "arch_hidden": FIXED_HIDDEN,
                },
                f,
                indent=2,
            )

        trial.set_user_attr("run_dir", str(run_dir))
        return float(winrate)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
    )
    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    summary = {
        "best_value": best.value,
        "best_params": best.params,
        "best_trial": best.number,
        "best_run_dir": best.user_attrs.get("run_dir"),
        "arch": {"hidden": FIXED_HIDDEN},
    }
    (pathlib.Path("runs") / "optuna_best.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nBest trial:")
    print(summary)


if __name__ == "__main__":
    main()
