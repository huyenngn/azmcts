import pathlib


def ensure_dir(p: pathlib.Path) -> None:
  """Create directory and parents if they don't exist."""
  p.mkdir(parents=True, exist_ok=True)
