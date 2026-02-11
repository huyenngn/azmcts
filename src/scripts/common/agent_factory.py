from __future__ import annotations

import typing as t

import numpy as np

import agents
import nets
from belief import samplers
from scripts.common import config, seeding
from utils import softmax

if t.TYPE_CHECKING:
  import random

  import openspiel


def make_belief_sampler(
  *,
  game: openspiel.Game,
  player_id: int,
  base_seed: int,
  sampler_cfg: config.SamplerConfig,
  run_id: str,
  purpose: str,
  game_idx: int | None,
  opponent_policy: samplers.OpponentPolicy | None = None,
) -> samplers.ParticleDeterminizationSampler:
  seed = seeding.derive_seed(
    base_seed,
    purpose=f"{purpose}/belief",
    run_id=run_id,
    game_idx=game_idx,
    player_id=player_id,
  )
  return samplers.ParticleDeterminizationSampler(
    game=game,
    ai_id=player_id,
    max_num_particles=sampler_cfg.max_num_particles,
    max_matches_per_particle=sampler_cfg.max_matches_per_particle,
    rebuild_tries=sampler_cfg.rebuild_tries,
    seed=seed,
    opponent_policy=opponent_policy,
  )


def make_agent(
  *,
  kind: str,
  player_id: int,
  game: openspiel.Game,
  search_cfg: config.SearchConfig,
  sampler_cfg: config.SamplerConfig,
  base_seed: int,
  run_id: str,
  purpose: str,
  device: str,
  model_path: str | None,
  net: nets.TinyPolicyValueNet
  | None = None,  # optional for training self-play
  game_idx: int | None,
) -> (
  tuple[agents.Agent, samplers.ParticleDeterminizationSampler]
  | tuple[None, None]
):
  if kind == "random":
    return None, None

  num_actions = game.num_distinct_actions()
  obs_size = game.observation_tensor_size()
  agent_seed = seeding.derive_seed(
    base_seed,
    purpose=f"{purpose}/agent",
    run_id=run_id,
    game_idx=game_idx,
    player_id=player_id,
  )

  opponent_policy: samplers.OpponentPolicy | None = None
  az_net = None

  if kind == "azbsmcts":
    if net is not None:
      az_net = net
    elif model_path is not None:
      az_net = nets.get_shared_az_model(
        obs_size=obs_size,
        num_actions=num_actions,
        model_path=model_path,
        device=device,
      )
    else:
      raise ValueError("azbsmcts requires either 'net' or 'model_path'")

    def make_opponent_policy(
      network: nets.TinyPolicyValueNet, dev: str
    ) -> samplers.OpponentPolicy:
      def policy_fn(states: list[openspiel.State]) -> list[np.ndarray]:
        import torch

        if not states:
          return []

        # Build batched observation tensor
        obs_list = []
        for state in states:
          side = state.current_player()
          obs = np.asarray(state.observation_tensor(side), dtype=np.float32)
          obs_list.append(obs)

        obs_batch = np.stack(obs_list, axis=0)
        obs_t = torch.from_numpy(obs_batch).to(dev)

        with torch.no_grad():
          logits_batch, _ = network(obs_t)
        logits_np = logits_batch.cpu().numpy()

        return [softmax.softmax_np(logits_np[i]) for i in range(len(states))]

      return policy_fn

    opponent_policy = make_opponent_policy(az_net, device)

  sampler = make_belief_sampler(
    game=game,
    player_id=player_id,
    base_seed=base_seed,
    sampler_cfg=sampler_cfg,
    run_id=run_id,
    purpose=purpose,
    game_idx=game_idx,
    opponent_policy=opponent_policy,
  )

  if kind == "bsmcts":
    return (
      agents.BSMCTSAgent(
        game=game,
        player_id=player_id,
        sampler=sampler,
        T=search_cfg.T,
        S=search_cfg.S,
        seed=agent_seed,
      ),
      sampler,
    )

  if kind == "azbsmcts":
    return (
      agents.AZBSMCTSAgent(
        game=game,
        player_id=player_id,
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
      sampler,
    )

  raise ValueError(f"Unknown agent kind: {kind}")


def select_action(
  kind: str,
  agent: agents.Agent | None,
  state: openspiel.State,
  rng: random.Random,
) -> int:
  if kind == "random" or agent is None:
    return rng.choice(state.legal_actions())
  return agent.select_action(state)
