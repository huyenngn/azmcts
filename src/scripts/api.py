"""Phantom Go API backend (single game at a time).

Constraints (by design):
- Only supports OpenSpiel "phantom_go" on 9x9 board.
- Single active game/session at a time.
- No environment variables for configuration.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import logging
import pathlib
import random
import re
import typing as t
from urllib import request

import fastapi
import pydantic
import uvicorn
from fastapi import responses

import openspiel
from scripts.common import agent_factory, config, seeding
from utils import utils

DEMO_MODEL_URL = (
  "https://github.com/huyenngn/azbsmcts/releases/download/demo-model/model.pt"
)
DEFAULT_DEMO_MODEL_PATH = pathlib.Path("models/demo_model.pt")

logger = logging.getLogger("phantom_go_api")

app = fastapi.FastAPI()

GAME_NAME = "phantom_go"
GAME_PARAMS = {"board_size": 9}


@dataclasses.dataclass
class ApiSettings:
  seed: int
  device: str
  search_cfg: config.SearchConfig
  sampler_cfg: config.SamplerConfig
  model_path: pathlib.Path


app.state.settings = None
app.state.game = None
app.state.state = None
app.state.human_id = 0
app.state.ai_id = 1
app.state.policy = "random"  # "random" | "bsmcts" | "azbsmcts"
app.state.agent = None
app.state.particle = None
app.state.rng = None
app.state.game_number = (
  0  # increments on /start so RNG changes deterministically between games
)


class PlayerColor(enum.Enum):
  Black = 0
  White = 1


class StartGameRequest(pydantic.BaseModel):
  player_id: int
  policy: str  # "random" | "bsmcts" | "azbsmcts"


class MakeMoveRequest(pydantic.BaseModel):
  action: int


class PreviousMoveInfo(pydantic.BaseModel):
  player: PlayerColor
  was_observational: bool = False
  was_pass: bool = False
  captured_stones: int = 0


class GameStateResponse(pydantic.BaseModel):
  current_player: PlayerColor
  observation: str = ""
  previous_move_info: PreviousMoveInfo | None = None
  is_terminal: bool = False
  returns: list[float] = []


class ParticlesResponse(pydantic.BaseModel):
  observations: list[str] = []
  diversity: float = 0.0


def _sse_event(event: str, data: dict[str, t.Any]) -> str:
  payload = json.dumps(data, separators=(",", ":"))
  sse = f"event: {event}\ndata: {payload}\n\n"
  logger.debug(f"Sending SSE event: {event}")
  return sse


def _ensure_model(path: pathlib.Path) -> str:
  """Download demo model if not present."""
  utils.ensure_dir(path.parent)
  if path.exists():
    return str(path)

  logger.info("Downloading demo model to %s", path)
  try:
    request.urlretrieve(DEMO_MODEL_URL, str(path))
  except Exception as e:
    raise fastapi.HTTPException(
      status_code=500, detail=f"Failed to download demo model: {e}"
    ) from e
  return str(path)


def _parse_move_info(player: PlayerColor) -> PreviousMoveInfo | None:
  """Parse last-move info from the observation string."""
  if app.state.state is None or app.state.state.is_terminal():
    return None

  obs_tail = app.state.state.observation_string(app.state.human_id)[-120:]
  m = re.search(
    r"Previous move was (valid|observational)"
    r"(?:\s+and was a (pass)|\s+In previous move (\d+) stones were captured)?",
    obs_tail,
  )
  if not m:
    return None

  return PreviousMoveInfo(
    player=player,
    was_observational=(m.group(1) == "observational"),
    was_pass=(m.group(2) is not None),
    captured_stones=int(m.group(3)) if m.group(3) else 0,
  )


def _apply_action(player: int, action: int) -> None:
  """Apply an action and update belief sampler (if any)."""
  app.state.state.apply_action(action)
  if app.state.particle is not None:
    app.state.particle.step(
      actor=player, action=action, real_state_after=app.state.state
    )


def _build_agent(policy: str) -> None:
  """Create agent + particle sampler for the current game."""
  settings: ApiSettings = app.state.settings

  if policy == "random":
    app.state.agent = None
    app.state.particle = None
    return

  model_path = None
  if policy == "azbsmcts":
    model_path = _ensure_model(settings.model_path)

  agent, particle = agent_factory.make_agent(
    kind=policy,
    player_id=app.state.ai_id,
    game=app.state.game,
    search_cfg=settings.search_cfg,
    sampler_cfg=settings.sampler_cfg,
    base_seed=settings.seed,
    run_id="api",
    purpose="api",
    device=settings.device,
    model_path=model_path,
    net=None,
    game_idx=app.state.game_number,  # deterministic variation across games
  )
  app.state.agent = agent
  app.state.particle = particle


def _play_ai_turns() -> t.Iterator[GameStateResponse]:
  """Execute AI turns and yield incremental responses."""
  while (
    app.state.state is not None
    and not app.state.state.is_terminal()
    and app.state.state.current_player() == app.state.ai_id
  ):
    action = agent_factory.select_action(
      app.state.policy, app.state.agent, app.state.state, app.state.rng
    )

    logger.info("AI plays action %d", action)
    _apply_action(app.state.ai_id, action)

    info = _parse_move_info(PlayerColor(app.state.ai_id))

    yield _game_state_response(info)


def _game_state_response(
  move_info: PreviousMoveInfo | None,
) -> GameStateResponse:
  st = app.state.state
  return GameStateResponse(
    current_player=PlayerColor(st.current_player()),
    observation=st.observation_string(app.state.human_id)
    if not st.is_terminal()
    else "",
    previous_move_info=move_info,
    is_terminal=st.is_terminal(),
    returns=list(st.returns()),
  )


@app.get("/")
def root() -> dict[str, t.Any]:
  return {
    "name": "Phantom Go API",
    "active_game": app.state.state is not None
    and not app.state.state.is_terminal(),
  }


def _step_stream(action: int) -> t.Iterator[str]:
  try:
    logger.info("Human plays action %d", action)
    _apply_action(app.state.human_id, action)

    info = _parse_move_info(PlayerColor(app.state.human_id))

    yield _sse_event(
      "update", _game_state_response(info).model_dump(mode="json")
    )

    for res in _play_ai_turns():
      yield _sse_event("update", res.model_dump(mode="json"))

    yield _sse_event(
      "done", _game_state_response(None).model_dump(mode="json")
    )
  except Exception as exc:
    logger.exception("Streaming step failed")
    yield _sse_event(
      "error",
      {"detail": str(exc)},
    )


def _start_game_stream() -> t.Iterator[str]:
  try:
    for res in _play_ai_turns():
      yield _sse_event("update", res.model_dump(mode="json"))
  except Exception as exc:
    logger.exception("Streaming start_game failed")
    yield _sse_event(
      "error",
      {"detail": str(exc)},
    )


@app.post("/start")
def start_game(request: StartGameRequest) -> responses.StreamingResponse:
  if request.player_id not in (0, 1):
    raise fastapi.HTTPException(
      status_code=400, detail="player_id must be 0 or 1"
    )
  if request.policy not in ("random", "bsmcts", "azbsmcts"):
    raise fastapi.HTTPException(
      status_code=400, detail="policy must be random|bsmcts|azbsmcts"
    )

  app.state.game_number += 1
  app.state.game = openspiel.Game(GAME_NAME, GAME_PARAMS)
  app.state.state = app.state.game.new_initial_state()

  app.state.human_id = request.player_id
  app.state.ai_id = 1 - request.player_id
  app.state.policy = request.policy

  settings: ApiSettings = app.state.settings
  rng_seed = seeding.derive_seed(
    settings.seed,
    purpose="api/rng",
    run_id="api",
    game_idx=app.state.game_number,
    player_id=app.state.ai_id,
  )

  app.state.rng = random.Random(rng_seed)

  logger.info(
    "Start game #%d: human=%d ai=%d policy=%s seed=%d T=%d S=%d",
    app.state.game_number,
    app.state.human_id,
    app.state.ai_id,
    app.state.policy,
    settings.seed,
    settings.search_cfg.T,
    settings.search_cfg.S,
  )

  _build_agent(app.state.policy)

  return responses.StreamingResponse(
    _start_game_stream(), media_type="text/event-stream"
  )


@app.post("/step")
def step(request: MakeMoveRequest) -> responses.StreamingResponse:
  st = app.state.state
  if st is None:
    raise fastapi.HTTPException(status_code=400, detail="No active game")
  if st.is_terminal():
    raise fastapi.HTTPException(status_code=400, detail="Game is over")
  if st.current_player() != app.state.human_id:
    raise fastapi.HTTPException(status_code=400, detail="Not your turn")
  if request.action not in st.legal_actions():
    raise fastapi.HTTPException(status_code=400, detail="Illegal action")

  return responses.StreamingResponse(
    _step_stream(request.action), media_type="text/event-stream"
  )


@app.get("/particles")
def get_particles(num_particles: int) -> ParticlesResponse:
  if app.state.game is None or app.state.particle is None:
    raise fastapi.HTTPException(status_code=400, detail="No active game")

  diversity = app.state.particle._get_particle_diversity()
  logger.info(f"Particle filter has {diversity:.1%} diversity")

  observations: list[str] = []
  particles = app.state.particle.sample_unique_particles(num_particles)
  for p in particles:
    obs = p.observation_string(app.state.human_id)
    observations.append(obs)

  return ParticlesResponse(observations=observations, diversity=diversity)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--host", type=str, default="0.0.0.0")
  p.add_argument("--port", type=int, default=8000)

  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", type=str, default="cpu")

  # Search (API defaults intentionally small)
  p.add_argument("--T", type=int, default=4)
  p.add_argument("--S", type=int, default=2)
  p.add_argument("--c-puct", type=float, default=1.5)
  p.add_argument("--dirichlet-alpha", type=float, default=0.0)
  p.add_argument("--dirichlet-weight", type=float, default=0.0)

  # Particle sampler
  p.add_argument("--max-num-particles", type=int, default=150)
  p.add_argument("--max-matches-per-particle", type=int, default=100)
  p.add_argument("--rebuild-tries", type=int, default=5)

  # Model path for azbsmcts
  p.add_argument(
    "--model-path", type=str, default=str(DEFAULT_DEMO_MODEL_PATH)
  )

  # Debug logging
  p.add_argument("--debug", action="store_true", help="Enable debug logging")

  args = p.parse_args()

  if args.debug:
    logging.basicConfig(level=logging.DEBUG)
  else:
    logging.basicConfig(level=logging.INFO)

  app.state.settings = ApiSettings(
    seed=args.seed,
    device=args.device,
    search_cfg=config.SearchConfig(
      T=args.T,
      S=args.S,
      c_puct=args.c_puct,
      dirichlet_alpha=args.dirichlet_alpha,
      dirichlet_weight=args.dirichlet_weight,
    ),
    sampler_cfg=config.SamplerConfig(
      max_num_particles=args.max_num_particles,
      max_matches_per_particle=args.max_matches_per_particle,
      rebuild_tries=args.rebuild_tries,
    ),
    model_path=pathlib.Path(args.model_path),
  )

  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
  main()
