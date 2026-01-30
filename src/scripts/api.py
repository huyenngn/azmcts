"""
Phantom Go API backend (single game at a time).

Constraints (by design):
- Only supports OpenSpiel "phantom_go" on 9x9 board.
- Single active game/session at a time.
- No environment variables for configuration.
"""

from __future__ import annotations

import argparse
import enum
import logging
import pathlib
import random
import re
import urllib.request
from dataclasses import dataclass

import fastapi
import pydantic
import pyspiel
import uvicorn

from scripts.common.agent_factory import make_agent, select_action
from scripts.common.config import SamplerConfig, SearchConfig
from scripts.common.seeding import derive_seed
from utils import utils

DEMO_MODEL_URL = "https://github.com/huyenngn/azbsmcts/releases/download/demo-model/model.pt"
DEFAULT_DEMO_MODEL_PATH = pathlib.Path("models/demo_model.pt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phantom_go_api")

app = fastapi.FastAPI()

GAME_NAME = "phantom_go"
GAME_PARAMS = {"board_size": 9}
MAX_REPEATED_ACTION = 3
PASS_ACTION = 81


@dataclass
class ApiSettings:
    seed: int
    device: str
    search_cfg: SearchConfig
    sampler_cfg: SamplerConfig
    model_path: pathlib.Path


app.state.settings = ApiSettings(
    seed=0,
    device="cpu",
    search_cfg=SearchConfig(
        T=4, S=2, c_puct=1.5, dirichlet_alpha=0.0, dirichlet_weight=0.0
    ),
    sampler_cfg=SamplerConfig(
        num_particles=32, opp_tries_per_particle=8, rebuild_max_tries=200
    ),
    model_path=DEFAULT_DEMO_MODEL_PATH,
)

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
    player: PlayerColor = PlayerColor.Black
    was_observational: bool = False
    was_pass: bool = False
    captured_stones: int = 0


class GameStateResponse(pydantic.BaseModel):
    observation: str = ""
    previous_move_infos: list[PreviousMoveInfo] = []
    is_terminal: bool = False
    returns: list[float] = []


def _ensure_model(path: pathlib.Path) -> str:
    """Download demo model if not present."""
    utils.ensure_dir(path.parent)
    if path.exists():
        return str(path)

    logger.info("Downloading demo model to %s", path)
    try:
        urllib.request.urlretrieve(DEMO_MODEL_URL, str(path))
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=500, detail=f"Failed to download demo model: {e}"
        )
    return str(path)


def _parse_move_info() -> PreviousMoveInfo | None:
    """Parse last-move info from the observation string."""
    if app.state.state is None or app.state.state.is_terminal():
        return None

    # OpenSpiel phantom_go observation includes a short "Previous move was ..." tail.
    obs_tail = app.state.state.observation_string(app.state.human_id)[-120:]
    m = re.search(
        r"Previous move was (valid|observational)"
        r"(?:\s+and was a (pass)|\s+In previous move (\d+) stones were captured)?",
        obs_tail,
    )
    if not m:
        return None

    return PreviousMoveInfo(
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

    agent, particle = make_agent(
        kind=policy,
        player_id=app.state.ai_id,
        game=app.state.game,
        search_cfg=settings.search_cfg,
        sampler_cfg=settings.sampler_cfg,
        base_seed=settings.seed,
        run_id="api",  # constant namespace, not a session concept
        purpose="api",
        device=settings.device,
        model_path=model_path,
        net=None,
        game_idx=app.state.game_number,  # deterministic variation across games
    )
    app.state.agent = agent
    app.state.particle = particle


def _play_ai_turns() -> list[PreviousMoveInfo]:
    """Execute all consecutive AI turns."""
    infos: list[PreviousMoveInfo] = []
    recent_actions: list[int] = []

    while (
        app.state.state is not None
        and not app.state.state.is_terminal()
        and app.state.state.current_player() == app.state.ai_id
    ):
        action = select_action(
            app.state.policy, app.state.agent, app.state.state, app.state.rng
        )

        if (
            action != PASS_ACTION
            and recent_actions.count(action) >= MAX_REPEATED_ACTION
        ):
            logger.warning(
                "AI stuck in loop (action %d repeated %d times), skipping action",
                action,
                recent_actions.count(action),
            )
            continue

        recent_actions.append(action)
        if len(recent_actions) > MAX_REPEATED_ACTION:
            recent_actions.pop(0)

        logger.info("AI plays action %d", action)
        _apply_action(app.state.ai_id, action)

        info = _parse_move_info()
        if info:
            info.player = PlayerColor(app.state.ai_id)
            infos.append(info)

    return infos


def _response(move_infos: list[PreviousMoveInfo]) -> GameStateResponse:
    st = app.state.state
    return GameStateResponse(
        observation=st.observation_string(app.state.human_id)
        if not st.is_terminal()
        else "",
        previous_move_infos=move_infos,
        is_terminal=st.is_terminal(),
        returns=list(st.returns()),
    )


@app.get("/")
def root():
    return {
        "name": "Phantom Go API",
        "active_game": app.state.state is not None
        and not app.state.state.is_terminal(),
    }


@app.post("/start")
def start_game(request: StartGameRequest) -> GameStateResponse:
    if request.player_id not in (0, 1):
        raise fastapi.HTTPException(
            status_code=400, detail="player_id must be 0 or 1"
        )
    if request.policy not in ("random", "bsmcts", "azbsmcts"):
        raise fastapi.HTTPException(
            status_code=400, detail="policy must be random|bsmcts|azbsmcts"
        )

    app.state.game_number += 1
    app.state.game = pyspiel.load_game(GAME_NAME, GAME_PARAMS)
    app.state.state = app.state.game.new_initial_state()

    app.state.human_id = request.player_id
    app.state.ai_id = 1 - request.player_id
    app.state.policy = request.policy

    settings: ApiSettings = app.state.settings
    rng_seed = derive_seed(
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
    move_infos = _play_ai_turns()
    return _response(move_infos)


@app.post("/step")
def step(request: MakeMoveRequest) -> GameStateResponse:
    st = app.state.state
    if st is None:
        raise fastapi.HTTPException(status_code=400, detail="No active game")
    if st.is_terminal():
        raise fastapi.HTTPException(status_code=400, detail="Game is over")
    if st.current_player() != app.state.human_id:
        raise fastapi.HTTPException(status_code=400, detail="Not your turn")
    if request.action not in st.legal_actions():
        raise fastapi.HTTPException(status_code=400, detail="Illegal action")

    logger.info("Human plays action %d", request.action)
    _apply_action(app.state.human_id, request.action)

    infos: list[PreviousMoveInfo] = []
    info = _parse_move_info()
    if info:
        info.player = PlayerColor(app.state.human_id)
        infos.append(info)

    infos.extend(_play_ai_turns())

    if st.is_terminal():
        logger.info("Game ended: returns=%s", st.returns())

    return _response(infos)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    # Explicit (no env vars)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")

    # Search knobs (API defaults intentionally small)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--S", type=int, default=2)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.0)
    p.add_argument("--dirichlet-weight", type=float, default=0.0)

    # Particle sampler knobs
    p.add_argument("--num-particles", type=int, default=32)
    p.add_argument("--opp-tries", type=int, default=8)
    p.add_argument("--rebuild-tries", type=int, default=200)

    # Model path for azbsmcts
    p.add_argument(
        "--model-path", type=str, default=str(DEFAULT_DEMO_MODEL_PATH)
    )

    args = p.parse_args()

    app.state.settings = ApiSettings(
        seed=args.seed,
        device=args.device,
        search_cfg=SearchConfig(
            T=args.T,
            S=args.S,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_weight=args.dirichlet_weight,
        ),
        sampler_cfg=SamplerConfig(
            num_particles=args.num_particles,
            opp_tries_per_particle=args.opp_tries,
            rebuild_max_tries=args.rebuild_tries,
        ),
        model_path=pathlib.Path(args.model_path),
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
