from __future__ import annotations

import typing as t

import pyspiel


class Game:
  """Wrapper around pyspiel.Game that returns wrapped State objects."""

  def __init__(
    self, name: str, params: dict[str, t.Any] | None = None
  ) -> None:
    self._game: pyspiel.Game = pyspiel.load_game(
      name, params if params is not None else {}
    )

  def new_initial_state(self) -> State:
    """Create a new initial game state."""
    return State(self._game.new_initial_state())

  def observation_tensor_size(self) -> int:
    """Return the size of observation tensors."""
    return self._game.observation_tensor_size()

  def num_distinct_actions(self) -> int:
    """Return the number of distinct actions."""
    return self._game.num_distinct_actions()

  def deserialize_state(self, serialized_state: str) -> State:
    """Deserialize a state from a string."""
    deserialized_state = self._game.deserialize_state(serialized_state)
    return State(deserialized_state)


class State:
  """Wrapper around pyspiel.State with attempted action tracking."""

  def __init__(
    self,
    state: pyspiel.State,
    attempted_actions: set[int] | None = None,
  ) -> None:
    self._state: pyspiel.State = state
    self._attempted_actions: set[int] = (
      attempted_actions if attempted_actions is not None else set()
    )

  def apply_action(self, action: int) -> None:
    """Apply an action to the state."""
    player = self._state.current_player()
    self._state.apply_action(action)
    if player == self._state.current_player():
      self._attempted_actions.add(action)
    else:
      self._attempted_actions.clear()

  def legal_actions(self) -> list[int]:
    """Return legal actions excluding already-attempted ones."""
    all_legal_actions = self._state.legal_actions()
    return [a for a in all_legal_actions if a not in self._attempted_actions]

  def current_player(self) -> int:
    """Return the current player."""
    return self._state.current_player()

  def is_terminal(self) -> bool:
    """Check if the game is over."""
    return self._state.is_terminal()

  def returns(self) -> list[float]:
    """Return the game returns for each player."""
    return self._state.returns()

  def observation_string(self, player: int) -> str:
    """Return observation string for a player."""
    return self._state.observation_string(player)

  def observation_tensor(self, player: int) -> list[float]:
    """Return observation tensor for a player."""
    return self._state.observation_tensor(player)

  def clone(self) -> State:
    """Create a deep copy of this state."""
    cloned_state = self._state.get_game().deserialize_state(
      self._state.serialize()
    )
    return State(cloned_state, set(self._attempted_actions))

  def game_length(self) -> int:
    """Return the length of the game so far."""
    return len(self._state.history())

  def serialize(self) -> str:
    """Return a string serialization of the state."""
    return self._state.serialize()
