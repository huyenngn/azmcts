from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class EdgeStats:
    """Statistics for a tree edge (action)."""

    n: int = 0
    w: float = 0.0
    p: float = 0.0

    @property
    def q(self) -> float:
        """Mean action value."""
        return 0.0 if self.n == 0 else self.w / self.n


@dataclasses.dataclass
class Node:
    """A node in the belief tree indexed by observation."""

    obs_key: str
    player_to_act: int
    is_expanded: bool = False
    edges: dict[int, EdgeStats] = dataclasses.field(default_factory=dict)
    legal_actions: list[int] = dataclasses.field(default_factory=list)
    n: int = 0

    def get_most_visited_action(self, actions: list[int] | None = None) -> int:
        """Return action with highest visit count."""
        if actions is None:
            actions = list(self.edges.keys())
        return max(actions, key=lambda a: self.edges.get(a, EdgeStats()).n)


class BeliefTree:
    """Tree structure mapping observations to nodes."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}

    def get_or_create(self, obs_key: str, player_to_act: int) -> Node:
        """Get existing node or create new one for observation."""
        if obs_key not in self._nodes:
            self._nodes[obs_key] = Node(
                obs_key=obs_key, player_to_act=player_to_act
            )
        return self._nodes[obs_key]
