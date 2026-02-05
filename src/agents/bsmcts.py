from __future__ import annotations

import math
import typing as t

import agents
from belief import samplers, tree

if t.TYPE_CHECKING:
  import openspiel


class BSMCTSAgent(agents.BaseAgent):
  """Belief-State Monte Carlo Tree Search agent.

  Uses determinization sampling to handle imperfect information.
  Does not use learned priors; relies on UCT for exploration.
  """

  def __init__(
    self,
    player_id: int,
    num_actions: int,
    sampler: samplers.DeterminizationSampler,
    c_uct: float = 1.4,
    T: int = 64,
    S: int = 8,
    seed: int = 0,
  ):
    super().__init__(player_id=player_id, num_actions=num_actions, seed=seed)
    self.tree = tree.BeliefTree()
    self.sampler = sampler
    self.c_uct = float(c_uct)
    self.T = int(T)
    self.S = int(S)

  def select_action(self, state: openspiel.State) -> int:
    """Select the best action via MCTS over sampled determinizations."""
    root = self.tree.get_or_create(
      self.obs_key(state, self.player_id), state.current_player()
    )

    if not root.is_expanded:
      self._expand(root, state)

    for _ in range(self.T):
      gamma = self.sampler.sample()
      for _ in range(self.S):
        self._search(gamma.clone())

    return root.get_most_visited_action()

  def _expand(self, node: tree.Node, state: openspiel.State) -> None:
    """Initialize node edges for all legal actions."""
    node.is_expanded = True
    node.legal_actions = list(state.legal_actions())
    for a in node.legal_actions:
      node.edges.setdefault(a, tree.EdgeStats())

  def _uct(self, parent: tree.Node, edge: tree.EdgeStats) -> float:
    q = edge.q
    u = self.c_uct * math.sqrt(math.log(parent.n + 1.0) / (edge.n + 1.0))
    return q + u

  def _rollout(self, state: openspiel.State) -> float:
    """Random playout to terminal or depth limit."""
    steps = 0
    while not state.is_terminal():
      la = state.legal_actions()
      if not la:
        break
      state.apply_action(self.rng.choice(la))
      steps += 1

    if state.is_terminal():
      return float(state.returns()[self.player_id])
    return 0.0

  def _search(self, state: openspiel.State) -> float:
    """Recursive MCTS search with UCT selection."""
    if state.is_terminal():
      return float(state.returns()[self.player_id])

    node = self.tree.get_or_create(
      self.obs_key(state, self.player_id), state.current_player()
    )
    node.n += 1

    if not node.is_expanded:
      self._expand(node, state)
      return self._rollout(state.clone())

    legal_now = set(state.legal_actions())
    best_a = None
    best_s = -1e18
    for a, e in node.edges.items():
      if a not in legal_now:
        continue
      s = self._uct(node, e)
      if s > best_s:
        best_s = s
        best_a = a

    if best_a is None:
      return self._rollout(state.clone())

    state.apply_action(best_a)
    v = self._search(state)

    edge = node.edges[best_a]
    edge.n += 1
    edge.w += v
    return v
