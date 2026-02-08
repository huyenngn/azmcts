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

  xs = np.array([r.get("x_games", float("nan")) for r in rows], dtype=float)
  wr_b = np.array(
    [r.get("wr_vs_bsmcts", float("nan")) for r in rows], dtype=float
  )
  wr_r = np.array(
    [r.get("wr_vs_random", float("nan")) for r in rows], dtype=float
  )

  # Extract game length data
  mean_win_b = np.array(
    [r.get("mean_win_len_vs_bsmcts", float("nan")) for r in rows], dtype=float
  )
  mean_loss_b = np.array(
    [r.get("mean_loss_len_vs_bsmcts", float("nan")) for r in rows], dtype=float
  )
  mean_draw_b = np.array(
    [r.get("mean_draw_len_vs_bsmcts", float("nan")) for r in rows], dtype=float
  )
  mean_win_r = np.array(
    [r.get("mean_win_len_vs_random", float("nan")) for r in rows], dtype=float
  )
  mean_loss_r = np.array(
    [r.get("mean_loss_len_vs_random", float("nan")) for r in rows], dtype=float
  )
  mean_draw_r = np.array(
    [r.get("mean_draw_len_vs_random", float("nan")) for r in rows], dtype=float
  )

  order = np.argsort(xs)
  xs = xs[order]
  wr_b, wr_r = wr_b[order], wr_r[order]
  mean_win_b = mean_win_b[order]
  mean_loss_b = mean_loss_b[order]
  mean_draw_b = mean_draw_b[order]
  mean_win_r = mean_win_r[order]
  mean_loss_r = mean_loss_r[order]
  mean_draw_r = mean_draw_r[order]

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

  # Plot game lengths by outcome
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  # Subplot 1: vs BS-MCTS
  ax1.plot(xs, mean_win_b, marker="o", label="Wins", color="green")
  ax1.plot(xs, mean_loss_b, marker="s", label="Losses", color="red")
  ax1.plot(xs, mean_draw_b, marker="^", label="Draws", color="gray")
  ax1.set_xlabel("Training games")
  ax1.set_ylabel("Average game length")
  ax1.set_title("Game Length vs BS-MCTS")
  ax1.legend()
  ax1.grid(True, alpha=0.3)

  # Subplot 2: vs Random
  ax2.plot(xs, mean_win_r, marker="o", label="Wins", color="green")
  ax2.plot(xs, mean_loss_r, marker="s", label="Losses", color="red")
  ax2.plot(xs, mean_draw_r, marker="^", label="Draws", color="gray")
  ax2.set_xlabel("Training games")
  ax2.set_ylabel("Average game length")
  ax2.set_title("Game Length vs Random")
  ax2.legend()
  ax2.grid(True, alpha=0.3)

  plt.suptitle(f"Game Length Analysis: {run_dir.name}")
  plt.tight_layout()

  out_path_lengths = run_dir / "eval_game_lengths.png"
  plt.savefig(out_path_lengths)
  plt.close()

  print(f"Wrote {out_path_lengths}")


if __name__ == "__main__":
  main()
