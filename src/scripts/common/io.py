from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import typing as t

from utils import utils


@dataclasses.dataclass(frozen=True)
class RunPaths:
    run_dir: pathlib.Path
    config_path: pathlib.Path
    train_metrics_path: pathlib.Path
    checkpoints_dir: pathlib.Path
    model_path: pathlib.Path
    eval_dir: pathlib.Path
    figures_dir: pathlib.Path


def make_run_dir(runs_root: str, run_name: str, seed: int) -> RunPaths:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = pathlib.Path(runs_root) / f"{ts}_{run_name}_seed{seed}"
    utils.ensure_dir(run_dir)

    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    figures_dir = run_dir / "figures"
    utils.ensure_dir(checkpoints_dir)
    utils.ensure_dir(eval_dir)
    utils.ensure_dir(figures_dir)

    return RunPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        train_metrics_path=run_dir / "train_metrics.jsonl",
        checkpoints_dir=checkpoints_dir,
        model_path=run_dir / "model.pt",
        eval_dir=eval_dir,
        figures_dir=figures_dir,
    )


def write_json(path: pathlib.Path, payload: t.Any) -> None:
    utils.ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(
    path: pathlib.Path, rows: t.Iterable[dict[str, t.Any]]
) -> None:
    utils.ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_json(path: pathlib.Path) -> t.Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, t.Any]]:
    rows: list[dict[str, t.Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows
