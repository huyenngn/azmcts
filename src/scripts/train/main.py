from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyspiel
import torch
from torch.nn import functional as F

import agents
import nets
from scripts.common import agent_factory, config, io, seeding
from utils import utils


def obs_tensor_side_to_move(state: pyspiel.State) -> np.ndarray:
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
    game: pyspiel.Game,
    net: nets.TinyPolicyValueNet,
    search_cfg: config.SearchConfig,
    sampler_cfg: config.SamplerConfig,
    base_seed: int,
    device: str,
    temperature: float,
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
            action, pi = a0.select_action_with_pi(
                state, temperature=temperature
            )
        else:
            action, pi = a1.select_action_with_pi(
                state, temperature=temperature
            )

        traj.append((obs, pi.astype(np.float32), p))

        actor = state.current_player()
        state.apply_action(action)

        # update both filters
        p0.step(actor=actor, action=action, real_state_after=state)
        p1.step(actor=actor, action=action, real_state_after=state)

    rets = state.returns()
    examples = [
        Example(obs=obs, pi=pi, z=float(rets[p])) for obs, pi, p in traj
    ]
    return examples, float(rets[0])


def self_play(
    *,
    game: pyspiel.Game,
    net: nets.TinyPolicyValueNet,
    num_games: int,
    search_cfg: config.SearchConfig,
    sampler_cfg: config.SamplerConfig,
    base_seed: int,
    device: str,
    temperature: float,
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
            temperature=temperature,
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
) -> None:
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temp", type=float, default=1.0)

    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="aztrain")
    p.add_argument("--out-model", type=str, default="")  # optional extra copy

    # sampler defaults (match your current eval hardcodes, but now explicit) :contentReference[oaicite:6]{index=6}
    p.add_argument("--num-particles", type=int, default=32)
    p.add_argument("--opp-tries", type=int, default=8)
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
        opp_tries_per_particle=args.opp_tries,
        rebuild_max_tries=args.rebuild_tries,
    )

    run = io.make_run_dir(args.runs_root, args.run_name, args.seed)
    run_id = run.run_dir.name

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
        temperature=args.temp,
        sampler=sampler_cfg,
    )

    # Write config with reproducibility fingerprint
    fingerprint = seeding.get_repro_fingerprint(args.device)
    config_payload = {
        "config": config.to_jsonable(cfg),
        "fingerprint": fingerprint.to_dict(),
    }
    io.write_json(run.config_path, config_payload)

    game = pyspiel.load_game(game_cfg.name, game_cfg.params)
    net = nets.TinyPolicyValueNet(
        obs_size=game.observation_tensor_size(),
        num_actions=game.num_distinct_actions(),
    ).to(args.device)

    # Checkpoints directory
    checkpoints_dir = run.run_dir / "checkpoints"
    utils.ensure_dir(checkpoints_dir)

    # Interleaved self-play and training (like real AlphaZero)
    interval = (
        args.checkpoint_interval
        if args.checkpoint_interval > 0
        else args.games
    )
    all_examples: list[Example] = []
    all_p0rets: list[float] = []
    games_played = 0
    checkpoint_idx = 0

    while games_played < args.games:
        # Determine how many games to play this iteration
        games_this_iter = min(interval, args.games - games_played)

        print(
            f"\n=== Self-play games {games_played + 1}-{games_played + games_this_iter} ==="
        )

        for gi in range(games_this_iter):
            game_examples, p0_ret = self_play_one_game(
                game=game,
                net=net,
                search_cfg=search_cfg,
                sampler_cfg=sampler_cfg,
                base_seed=args.seed,
                device=args.device,
                temperature=args.temp,
                run_id=run_id,
                game_idx=games_played + gi,
            )
            all_examples.extend(game_examples)
            all_p0rets.append(p0_ret)

        games_played += games_this_iter
        print(f"total examples: {len(all_examples)}, games: {games_played}")
        print(f"p0 return mean: {float(np.mean(all_p0rets)):.3f}")

        # Train on all accumulated data
        print(f"training on {len(all_examples)} examples...")
        train_net(
            net=net,
            examples=all_examples,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device=args.device,
            seed=seeding.derive_seed(
                args.seed,
                purpose="train/sgd",
                run_id=run_id,
                extra=str(games_played),
            ),
            metrics_path=run.train_metrics_path,
        )

        # Save checkpoint after this training iteration
        if args.checkpoint_interval > 0 or games_played == args.games:
            ckpt_path = (
                checkpoints_dir / f"checkpoint_games_{games_played:05d}.pt"
            )
            torch.save(net.state_dict(), str(ckpt_path))
            print(f"checkpoint: {ckpt_path}")
            checkpoint_idx += 1

    # Save final model as canonical checkpoint
    torch.save(net.state_dict(), str(run.model_path))
    print(f"\nfinal model: {run.model_path}")
    print(f"run dir: {run.run_dir}")
    print(f"checkpoints: {checkpoint_idx}")

    if args.out_model:
        outp = pathlib.Path(args.out_model)
        utils.ensure_dir(outp.parent)
        torch.save(net.state_dict(), str(outp))
        print(f"also saved: {outp}")


if __name__ == "__main__":
    main()
