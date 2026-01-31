# azbsmcts

![License: Apache 2.0](https://img.shields.io/github/license/huyenngn/azbsmcts)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![CI](https://github.com/huyenngn/azbsmcts/actions/workflows/ci.yml/badge.svg?branch=main)
![Smoke](https://github.com/huyenngn/azbsmcts/actions/workflows/smoke.yml/badge.svg?branch=main)
![Test](https://github.com/huyenngn/azbsmcts/actions/workflows/test.yml/badge.svg?branch=main)
![Demo model downloads](https://img.shields.io/github/downloads/huyenngn/azbsmcts/demo-model/model.pt?label=demo%20model%20downloads)

AlphaZero-inspired AI agent for **Phantom Go** (imperfect-information Go), developed as part of a research thesis.

The core contribution of this project is **AZ-BS-MCTS**: an AlphaZero-guided Belief-State Monte Carlo Tree Search algorithm that operates under strict partial-information constraints.

For game simulation and environment handling, this project builds upon [OpenSpiel](https://github.com/google-deepmind/open_spiel).

## Design Principles

This codebase intentionally enforces the following rules:

- **No access to hidden state**
  - Determinization is performed via belief-state particle filtering.
  - Cloning the real environment state is explicitly disallowed.
- **Explicit player perspectives**
  - `observation_tensor(player_id)` and `observation_string(player_id)` are always used.
  - Ambiguous default observation calls are never used.
- **Clear value semantics**
  - Neural network values are defined from the _side-to-move_ perspective.
  - Values are converted to a fixed root-player perspective during tree backup.
- **Reproducibility over convenience**
  - All scripts accept explicit random seeds.
  - Results are statistically reproducible (not bitwise deterministic), which is standard for MCTS-based methods.

## Demo

Use [uv](https://docs.astral.sh/uv/) to set up a local development environment.

```sh
git clone https://github.com/huyenngn/azbsmcts.git
cd azbsmcts
uv sync
```

You can use `uv run <command>` to avoid having to manually activate the project
venv. For example, to start the backend for the demo, run:

```sh
uv run api
```

For the frontend, follow instructions in [`demo`](demo).

## Scripts Overview

All functionality is exposed via `uv run <script>` commands,
as defined in `pyproject.toml`:

- `api` – Start FastAPI backend for interactive play (demo only)
- `train` – Run self-play training and log results to `runs/`
- `tune` – Optuna-based hyperparameter tuning
- `eval-match` – Single match evaluation between two agents
- `eval-sweep` – Sweep evaluation across multiple checkpoints
- `plot-train` – Generate plots from training logs
- `plot-tune` – Generate plots from tuning results
- `plot-eval` – Generate plots from evaluation results

## Experimental Workflow

### 1. Hyperparameter Tuning (Optional)

Optuna-based tuning for MCTS and learning parameters.
This is intended for **exploratory tuning** to verify that learning proceeds
and search remains stable.

```sh
uv run tune \
  --trials 50 \
  --games 100 \
  --epochs 5 \
  --eval-n 30 \
  --seed 42 \
  --storage sqlite:///runs/optuna.db \
  --study-name az_explore
```

**Tuned parameters:**

| Parameter         | Range       | Description                       |
| ----------------- | ----------- | --------------------------------- |
| `T`               | 2–16        | MCTS iterations per move          |
| `S`               | 2–8         | Belief samples (determinizations) |
| `c_puct`          | 0.5–3.0     | PUCT exploration constant         |
| `lr`              | 1e-4 – 3e-3 | Learning rate                     |
| `temp`            | 0.5–1.5     | Action sampling temperature       |
| `dirichlet_alpha` | 0.01–0.5    | Root exploration noise            |
| `num_particles`   | 10–64       | Belief state particles            |

Each trial produces its own directory under `runs/`. Best trial summary is written to `runs/optuna_best.json`.

---

### 2. Training

Run self-play training with parameters from tuning or manual selection:

```sh
uv run train \
  --games 1000 \
  --checkpoint-interval 100 \
  --T 8 \
  --S 4 \
  --c-puct 1.5 \
  --dirichlet-alpha 0.03 \
  --num-particles 32 \
  --epochs 5 \
  --batch 64 \
  --lr 1e-3 \
  --temp 1.0 \
  --device cuda \
  --seed 42
```

Training uses **interleaved self-play and learning** (like AlphaZero):

1. Play `--checkpoint-interval` games using current network
2. Train on all accumulated data
3. Save checkpoint, repeat until `--games` reached

Each training run creates a directory under `runs/` containing:

- `config.json` – full configuration and reproducibility fingerprint
- `train_metrics.jsonl` – per-iteration training losses
- `model.pt` – final trained network
- `checkpoints/` – intermediate checkpoints at each interval
  - `checkpoint_games_00100.pt`, `checkpoint_games_00200.pt`, etc.

> **Scaling study:** Use `--checkpoint-interval 100` with `--games 1000` to get
> 10 checkpoints (at 100, 200, ..., 1000 games) in a single run. Then use
> `eval-sweep` to evaluate all checkpoints.

---

### 3. Evaluation

#### Single Match

Evaluate one model against a baseline:

```sh
uv run eval-match \
  --model runs/<run_dir>/model.pt \
  --a azbsmcts \
  --b bsmcts \
  --n 50 \
  --T 8 \
  --S 4 \
  --c-puct 1.5 \
  --seed 42 \
  --out-json runs/<run_dir>/eval.json
```

#### Checkpoint Sweep

Evaluate all checkpoints from a single training run:

```sh
uv run eval-sweep \
  --run-dir runs/<run_dir> \
  --n 20 \
  --T 8 \
  --S 4 \
  --c-puct 1.5 \
  --seed 42
```

This generates `runs/<run_dir>/eval_sweep.jsonl` with win rates for each checkpoint.

---

### 4. Plotting

#### Training Loss Curves

```sh
uv run plot-train --run-dir runs/<run_dir>
```

Outputs to `runs/<run_dir>/`:

- `train_loss.png` – loss curves (total, policy, value)

#### Hyperparameter Analysis

```sh
uv run plot-tune --study-name az_explore
```

Outputs to `plots/tune/`:

- `optimization_history.png` – objective value over trials
- `param_importances.png` – which hyperparameters matter most
- `parallel_coordinate.png` – all params → objective in one view
- `slice_plots.png` – objective vs each parameter individually
- `tune_summary.json` – best params and value

#### Evaluation Win Rates

```sh
uv run plot-eval --run-dir runs/<run_dir>
```

Outputs to `runs/<run_dir>/`:

- `eval_winrate.png` – win rate vs training budget for that run

## Reproducibility

All scripts support explicit random seeds via `--seed`. Seeds are derived deterministically using `derive_seed()` with namespacing to ensure:

- Same base seed + same config → identical self-play action sequences
- Same base seed + same config → identical evaluation match outcomes

For maximum reproducibility, use `--deterministic-torch` (may reduce GPU performance).

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for detailed guarantees, limitations, and best practices.

## License

This project is licensed under the Apache License 2.0. For the full license text, see the [`LICENSE`](LICENSE) file.

It contains modifications of [OpenSpiel's](https://github.com/google-deepmind/open_spiel) AlphaZero algorithm and MCTS implementations, originally developed by Google DeepMind. The original license has been preserved in the relevant source files.
