"""Generate plots from Optuna hyperparameter tuning results."""

from __future__ import annotations

import argparse
import json
import pathlib
import typing as t

import optuna
from matplotlib import figure
from matplotlib import pyplot as plt
from optuna.visualization import matplotlib as optuna_plot

from utils import utils

if t.TYPE_CHECKING:
    from matplotlib import axes


def _save_plot(
    ax: axes.Axes, path: pathlib.Path, width: float = 10, height: float = 6
) -> None:
    """Save a plot from an Axes object."""
    fig = ax.get_figure()
    if fig is None or not isinstance(fig, figure.Figure):
        return
    fig.set_size_inches(width, height)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate plots from Optuna tuning results"
    )
    p.add_argument(
        "--storage",
        type=str,
        default="sqlite:///runs/optuna.db",
        help="Optuna storage URL",
    )
    p.add_argument(
        "--study-name",
        type=str,
        default="az_tune",
        help="Name of the study to plot",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="plots/tune",
        help="Output directory for plots",
    )
    p.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format",
    )
    args = p.parse_args()

    outdir = pathlib.Path(args.outdir)
    utils.ensure_dir(outdir)

    # Load study
    try:
        study = optuna.load_study(
            study_name=args.study_name, storage=args.storage
        )
    except KeyError:
        print(f"Study '{args.study_name}' not found in {args.storage}")
        print("Available studies:")
        summaries = optuna.get_all_study_summaries(storage=args.storage)
        for s in summaries:
            print(f"  - {s.study_name} ({s.n_trials} trials)")
        return

    n_trials = len(study.trials)
    if n_trials == 0:
        print(f"Study '{args.study_name}' has no completed trials")
        return

    print(f"Loaded study '{args.study_name}' with {n_trials} trials")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # 1. Optimization history
    ax = optuna_plot.plot_optimization_history(study)
    _save_plot(ax, outdir / f"optimization_history.{args.format}")

    # 2. Parameter importances (requires >= 2 completed trials)
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed) >= 2:
        try:
            ax = optuna_plot.plot_param_importances(study)
            _save_plot(ax, outdir / f"param_importances.{args.format}")
        except Exception as e:
            print(f"Could not plot param importances: {e}")
    else:
        print("Skipping param importances (need >= 2 completed trials)")

    # 3. Parallel coordinate plot
    try:
        ax = optuna_plot.plot_parallel_coordinate(study)
        _save_plot(ax, outdir / f"parallel_coordinate.{args.format}", width=12)
    except Exception as e:
        print(f"Could not plot parallel coordinate: {e}")

    # 4. Slice plots (one per parameter)
    try:
        ax = optuna_plot.plot_slice(study)
        _save_plot(
            ax, outdir / f"slice_plots.{args.format}", width=14, height=10
        )
    except Exception as e:
        print(f"Could not plot slice: {e}")

    # 5. Summary stats to JSON
    summary = {
        "study_name": args.study_name,
        "n_trials": n_trials,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "direction": study.direction.name,
    }

    summary_path = outdir / "tune_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print(f"\nAll plots written to {outdir}/")


if __name__ == "__main__":
    main()
