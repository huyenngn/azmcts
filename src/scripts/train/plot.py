from __future__ import annotations

import argparse
import pathlib

from matplotlib import pyplot as plt

from scripts.common import io


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a single training run directory (e.g., runs/run_20260130_123456)",
    )
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    metrics_path = run_dir / "train_metrics.jsonl"

    if not metrics_path.exists():
        print(f"No train_metrics.jsonl found in {run_dir}")
        return

    rows = io.read_jsonl(metrics_path)
    if not rows:
        print(f"No rows in {metrics_path}")
        return

    epochs = [r["epoch"] for r in rows]
    loss = [r["loss"] for r in rows]

    plt.figure()
    plt.plot(epochs, loss, label="Total loss")

    if all("policy_loss" in r for r in rows):
        plt.plot(
            epochs,
            [float(r["policy_loss"]) for r in rows],
            label="Policy loss",
        )
    if all("value_loss" in r for r in rows):
        plt.plot(
            epochs, [float(r["value_loss"]) for r in rows], label="Value loss"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training: {run_dir.name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = run_dir / "train_loss.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
