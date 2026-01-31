from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from scripts.common import io


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a single training run directory (e.g., runs/run_20260130_123456)",
    )
    p.add_argument("--x", type=str, default="x_games")
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    inp = run_dir / "eval_sweep.jsonl"

    if not inp.exists():
        print(f"No eval_sweep.jsonl found in {run_dir}")
        print("Run 'eval-sweep --run-dir <run>' first.")
        return

    rows = io.read_jsonl(inp)
    if not rows:
        print(f"No rows in {inp}")
        return

    xs = np.array([r.get(args.x, float("nan")) for r in rows], dtype=float)
    wr_b = np.array(
        [r.get("wr_vs_bsmcts", float("nan")) for r in rows], dtype=float
    )
    wr_r = np.array(
        [r.get("wr_vs_random", float("nan")) for r in rows], dtype=float
    )

    order = np.argsort(xs)
    xs, wr_b, wr_r = xs[order], wr_b[order], wr_r[order]

    plt.figure()
    plt.plot(xs, wr_b, marker="o", label="AZ vs BS-MCTS")
    plt.plot(xs, wr_r, marker="s", label="AZ vs Random")
    plt.xlabel("Training games")
    plt.ylabel("Win rate")
    plt.ylim(0.0, 1.0)
    plt.title(f"Evaluation: {run_dir.name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = run_dir / "eval_winrate.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
