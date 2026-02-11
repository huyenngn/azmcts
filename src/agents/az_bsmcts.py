from __future__ import annotations

import math
import typing as t

import numpy as np
import torch

import nets
from agents import base
from belief import samplers, tree
from utils import softmax

if t.TYPE_CHECKING:
  import openspiel


class AZBSMCTSAgent(base.BaseAgent, base.PolicyTargetMixin):
  """AlphaZero-guided Belief-State MCTS agent.

  Uses neural network priors and values with PUCT selection.
  All observations use explicit player IDs to prevent information leakage.
  """

  def __init__(
    self,
    game: openspiel.Game,
    player_id: int,
    obs_size: int,
    sampler: samplers.DeterminizationSampler,
    c_puct: float = 1.5,
    T: int = 100,
    S: int = 8,
    seed: int = 0,
    device: str = "cpu",
    net: nets.TinyPolicyValueNet | None = None,
    model_path: str | None = None,
    dirichlet_alpha: float = 0.03,
    dirichlet_weight: float = 0.25,
    length_discount: float = 0.999,
  ):
    super().__init__(game=game, player_id=player_id, seed=seed)
    self.tree = tree.BeliefTree()
    self.sampler = sampler
    self.c_puct = float(c_puct)
    self.T = int(T)
    self.S = int(S)
    self.dirichlet_alpha = float(dirichlet_alpha)
    self.dirichlet_weight = float(dirichlet_weight)
    self.length_discount = float(length_discount)

    self.device = device
    if net is None:
      if model_path is None:
        raise ValueError("Either 'net' or 'model_path' must be provided")
      self.net = nets.get_shared_az_model(
        obs_size=obs_size,
        num_actions=game.num_distinct_actions(),
        model_path=model_path,
        device=device,
      )
    else:
      self.net = net.to(device)
      self.net.eval()

    self._obs_size = int(obs_size)

    # Pending leaf evaluations for batching:
    # (node, state, needs_expand, path from root to this leaf)
    self._pending_leaves: list[
      tuple[
        tree.Node,
        openspiel.State,
        bool,
        list[tuple[tree.Node, int]],
      ]
    ] = []

  def _state_tensor_side_to_move(self, state: openspiel.State) -> torch.Tensor:
    """Get observation tensor from current player's perspective."""
    side = state.current_player()
    obs = self.obs_tensor(state, side)
    if obs.size != self._obs_size:
      raise ValueError(
        f"obs_size mismatch: expected {self._obs_size}, got {obs.size}"
      )
    return torch.from_numpy(obs).to(self.device)

  def _expand(
    self,
    node: tree.Node,
    state: openspiel.State,
    add_dirichlet: bool = False,
  ) -> None:
    """Expand node and initialize edges with network priors.

    Args:
        node: The node to expand.
        state: The game state at this node.
        add_dirichlet: If True, add Dirichlet noise to priors (for root).
    """
    node.is_expanded = True
    node.legal_actions = list(state.legal_actions())
    for a in node.legal_actions:
      node.edges.setdefault(a, tree.EdgeStats())

    with torch.no_grad():
      x = self._state_tensor_side_to_move(state).unsqueeze(0)
      logits, _v = self.net(x)
      logits = logits.squeeze(0).detach().cpu().numpy()

    mask = np.full((self.game.num_distinct_actions(),), -1e9, dtype=np.float32)
    for a in node.legal_actions:
      mask[a] = 0.0
    priors = softmax.softmax_np(logits + mask)

    if add_dirichlet and self.dirichlet_alpha > 0:
      dir_seed = self.rng.getrandbits(32)
      dir_rng = np.random.default_rng(dir_seed)
      noise = dir_rng.dirichlet(
        [self.dirichlet_alpha] * len(node.legal_actions)
      )
      eps = self.dirichlet_weight
      for i, a in enumerate(node.legal_actions):
        priors[a] = (1 - eps) * priors[a] + eps * noise[i]

    for a in node.legal_actions:
      node.edges[a].p = float(priors[a])

  def _leaf_value_root_perspective(
    self, state: openspiel.State, value: float | None = None
  ) -> float:
    """Get network value and convert to root player's perspective.

    Args:
        state: The game state at the leaf.
        value: Pre-computed value from batch evaluation. If None, runs inference.
    """
    if value is None:
      with torch.no_grad():
        x = self._state_tensor_side_to_move(state).unsqueeze(0)
        _, v = self.net(x)
        value = float(v.item())
    return value if state.current_player() == self.player_id else -value

  def _evaluate_pending_leaves(self) -> None:
    """Batch evaluate all pending leaf nodes and backpropagate values."""
    if not self._pending_leaves:
      return

    # Build batch tensor
    tensors = [
      self._state_tensor_side_to_move(state)
      for _, state, _, _ in self._pending_leaves
    ]
    batch = torch.stack(tensors, dim=0)

    with torch.no_grad():
      logits_batch, values_batch = self.net(batch)
      logits_batch = logits_batch.detach().cpu().numpy()
      values_batch = values_batch.detach().cpu().numpy()

    # Apply results to each pending leaf
    for i, (node, state, needs_expand, path) in enumerate(
      self._pending_leaves
    ):
      # Expand the leaf node if needed
      if needs_expand and not node.is_expanded:
        node.is_expanded = True
        node.legal_actions = list(state.legal_actions())
        for a in node.legal_actions:
          node.edges.setdefault(a, tree.EdgeStats())

        logits = logits_batch[i]
        mask = np.full(
          (self.game.num_distinct_actions(),), -1e9, dtype=np.float32
        )
        for a in node.legal_actions:
          mask[a] = 0.0
        priors = softmax.softmax_np(logits + mask)

        for a in node.legal_actions:
          node.edges[a].p = float(priors[a])

      # Backpropagate the real value along the stored path
      v = self._leaf_value_root_perspective(
        state, value=float(values_batch[i].item())
      )
      # Update the leaf node itself
      node.n += 1
      # Walk back up the path and update each ancestor
      for ancestor, action in path:
        ancestor.n += 1
        edge = ancestor.edges[action]
        edge.n += 1
        edge.w += v

    self._pending_leaves.clear()

  def _puct(self, parent: tree.Node, edge: tree.EdgeStats) -> float:
    q = edge.q
    u = self.c_puct * edge.p * math.sqrt(parent.n + 1.0) / (1.0 + edge.n)
    return q + u

  def _search(
    self,
    state: openspiel.State,
    batch_mode: bool = False,
    _path: list[tuple[tree.Node, int]] | None = None,
  ) -> float | None:
    """Recursive MCTS search with PUCT selection and NN evaluation.

    Args:
        state: Current game state (will be mutated during search).
        batch_mode: If True, queue leaf for batch evaluation instead of
            immediate inference. Returns None (deferred).
        _path: Internal — ancestors visited so far for deferred backprop.

    Returns:
        The leaf value from root's perspective, or None if deferred.
    """
    if state.is_terminal():
      # Apply length discount to discourage stalling
      return float(state.returns()[self.player_id]) * (
        self.length_discount ** state.game_length()
      )

    node = self.tree.get_or_create(
      self.obs_key(state, self.player_id), state.current_player()
    )

    if not node.is_expanded:
      if batch_mode:
        # Queue for batch evaluation with the path for deferred backprop
        self._pending_leaves.append((node, state, True, list(_path or [])))
        return None  # Sentinel: do not backpropagate yet
      node.n += 1
      self._expand(node, state)
      return self._leaf_value_root_perspective(state)

    legal_now = set(state.legal_actions())
    best_a = None
    best_s = -1e18
    for a, e in node.edges.items():
      if a not in legal_now:
        continue
      s = self._puct(node, e)
      if s > best_s:
        best_s = s
        best_a = a

    if best_a is None:
      if batch_mode:
        self._pending_leaves.append((node, state, False, list(_path or [])))
        return None
      node.n += 1
      return self._leaf_value_root_perspective(state)

    # Build path for potential deferred backprop
    if batch_mode:
      child_path = list(_path or [])
      child_path.append((node, best_a))
    else:
      child_path = None

    state.apply_action(best_a)
    v_root = self._search(state, batch_mode=batch_mode, _path=child_path)

    if v_root is None:
      # Child was deferred — do not backpropagate placeholder
      return None

    # Normal backpropagation with real value
    node.n += 1
    edge = node.edges[best_a]
    edge.n += 1
    edge.w += v_root
    return v_root

  def _root_visit_policy(
    self, root: tree.Node, temperature: float
  ) -> np.ndarray:
    pi = np.zeros((self.game.num_distinct_actions(),), dtype=np.float32)
    if not root.edges:
      return pi

    actions = list(root.edges.keys())
    visits = np.array([root.edges[a].n for a in actions], dtype=np.float32)

    if temperature <= 1e-8:
      pi[actions[int(np.argmax(visits))]] = 1.0
      return pi

    vt = np.power(visits + 1e-8, 1.0 / float(temperature))
    probs = vt / float(np.sum(vt))
    for a, p in zip(actions, probs, strict=False):
      pi[a] = float(p)
    return pi

  def select_action(self, state: openspiel.State) -> int:
    """Select best action (greedy, no exploration noise).

    Used during evaluation. No Dirichlet noise, argmax over visits.
    """
    a, _ = self._select_action_impl(
      state, temperature=1e-8, add_dirichlet=False
    )
    return a

  def select_action_with_pi(
    self, state: openspiel.State
  ) -> tuple[int, np.ndarray]:
    """Select action with policy vector for training.

    Used during self-play. Adds Dirichlet noise and samples from visits.
    """
    # Favor exploration for the first 20 plies, then switch to greedy
    temperature = 1.0 if state.game_length() < 20 else 1e-8
    return self._select_action_impl(
      state, temperature=temperature, add_dirichlet=True
    )

  def _select_action_impl(
    self,
    state: openspiel.State,
    temperature: float,
    add_dirichlet: bool,
  ) -> tuple[int, np.ndarray]:
    """Core action selection logic.

    Args:
        state: Current game state.
        temperature: Sampling temperature (0 = greedy).
        add_dirichlet: Whether to add Dirichlet noise at root.

    Returns
    -------
        Tuple of (action, policy vector).
    """
    root = self.tree.get_or_create(
      self.obs_key(state, self.player_id), state.current_player()
    )
    if not root.is_expanded:
      self._expand(root, state, add_dirichlet=add_dirichlet)

    for _ in range(self.T):
      gamma = self.sampler.sample()
      for _ in range(self.S):
        self._search(self.game.deserialize_state(gamma), batch_mode=True)
      # Batch evaluate all leaves collected in this iteration
      self._evaluate_pending_leaves()

    pi = self._root_visit_policy(root, temperature=float(temperature))

    legal = set(state.legal_actions())
    probs = pi.copy()
    for a in range(self.game.num_distinct_actions()):
      if a not in legal:
        probs[a] = 0.0

    s = float(np.sum(probs))
    if s <= 0:
      # No legal moves had probability - fallback to most visited
      best_a = root.get_most_visited_action(actions=list(legal))
      return int(best_a), pi

    probs /= s
    r = self.rng.random()
    cum = 0.0
    action = 0
    for a in range(self.game.num_distinct_actions()):
      pa = float(probs[a])
      if pa <= 0.0:
        continue
      cum += pa
      if r <= cum:
        action = a
        break
    return int(action), pi
