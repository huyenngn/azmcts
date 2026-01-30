from __future__ import annotations

import random

import numpy as np
import pyspiel
import torch

from agents import AZBSMCTSAgent, BSMCTSAgent
from belief.samplers.particle import (
    OpponentPolicy,
    ParticleBeliefSampler,
    ParticleDeterminizationSampler,
)
from scripts.common.config import SamplerConfig, SearchConfig
from scripts.common.seeding import derive_seed
from utils.softmax import softmax_np


def make_belief_sampler(
    *,
    game: pyspiel.Game,
    player_id: int,
    base_seed: int,
    sampler_cfg: SamplerConfig,
    run_id: str,
    purpose: str,
    game_idx: int | None,
    opponent_policy: OpponentPolicy | None = None,
) -> ParticleBeliefSampler:
    seed = derive_seed(
        base_seed,
        purpose=f"{purpose}/belief",
        run_id=run_id,
        game_idx=game_idx,
        player_id=player_id,
    )
    return ParticleBeliefSampler(
        game=game,
        ai_id=player_id,
        num_particles=sampler_cfg.num_particles,
        opp_tries_per_particle=sampler_cfg.opp_tries_per_particle,
        rebuild_max_tries=sampler_cfg.rebuild_max_tries,
        seed=seed,
        opponent_policy=opponent_policy,
    )


def make_agent(
    *,
    kind: str,
    player_id: int,
    game: pyspiel.Game,
    search_cfg: SearchConfig,
    sampler_cfg: SamplerConfig,
    base_seed: int,
    run_id: str,
    purpose: str,
    device: str,
    model_path: str | None,
    net=None,  # optional for training self-play
    game_idx: int | None,
):
    if kind == "random":
        return None, None

    num_actions = game.num_distinct_actions()
    obs_size = game.observation_tensor_size()
    agent_seed = derive_seed(
        base_seed,
        purpose=f"{purpose}/agent",
        run_id=run_id,
        game_idx=game_idx,
        player_id=player_id,
    )

    # For AZ-BS-MCTS, load/use network and create opponent policy
    opponent_policy: OpponentPolicy | None = None
    az_net = None

    if kind == "azbsmcts":
        from nets.tiny_policy_value import (
            TinyPolicyValueNet,
            get_shared_az_model,
        )

        if net is not None:
            az_net = net
        elif model_path is not None:
            az_net = get_shared_az_model(
                obs_size=obs_size,
                num_actions=num_actions,
                model_path=model_path,
                device=device,
            )
        else:
            raise ValueError("azbsmcts requires either 'net' or 'model_path'")

        # Create opponent policy using the network
        def make_opponent_policy(
            network: TinyPolicyValueNet, dev: str, osize: int
        ) -> OpponentPolicy:
            def policy_fn(state: pyspiel.State) -> np.ndarray:
                # Get observation from opponent's perspective (side to move)
                side = state.current_player()
                obs = np.asarray(
                    state.observation_tensor(side), dtype=np.float32
                )
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(dev)
                with torch.no_grad():
                    logits, _ = network(obs_t)
                # Convert to probabilities using numpy softmax (consistent with az_bsmcts.py)
                logits_np = logits.squeeze(0).cpu().numpy()
                return softmax_np(logits_np)

            return policy_fn

        opponent_policy = make_opponent_policy(az_net, device, obs_size)

    # Create belief sampler with opponent policy (if available)
    particle = make_belief_sampler(
        game=game,
        player_id=player_id,
        base_seed=base_seed,
        sampler_cfg=sampler_cfg,
        run_id=run_id,
        purpose=purpose,
        game_idx=game_idx,
        opponent_policy=opponent_policy,
    )
    sampler = ParticleDeterminizationSampler(particle)

    if kind == "bsmcts":
        return (
            BSMCTSAgent(
                player_id=player_id,
                num_actions=num_actions,
                sampler=sampler,
                T=search_cfg.T,
                S=search_cfg.S,
                seed=agent_seed,
            ),
            particle,
        )

    if kind == "azbsmcts":
        return (
            AZBSMCTSAgent(
                player_id=player_id,
                num_actions=num_actions,
                obs_size=obs_size,
                sampler=sampler,
                T=search_cfg.T,
                S=search_cfg.S,
                c_puct=search_cfg.c_puct,
                seed=agent_seed,
                device=device,
                model_path=None,  # Already loaded
                net=az_net,
                dirichlet_alpha=search_cfg.dirichlet_alpha,
                dirichlet_weight=search_cfg.dirichlet_weight,
            ),
            particle,
        )

    raise ValueError(f"Unknown agent kind: {kind}")


def select_action(
    kind: str, agent, state: pyspiel.State, rng: random.Random
) -> int:
    if kind == "random" or agent is None:
        return rng.choice(state.legal_actions())
    return agent.select_action(state)
