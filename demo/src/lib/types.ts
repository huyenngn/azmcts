export type StartGameRequest = {
  player_id: number
  policy: string
}

export type MakeMoveRequest = {
  action: number
}

export type GameStateResponse = {
  current_player: PlayerColor
  observation: string
  previous_move_info: PreviousMoveInfo
  is_terminal: boolean
  returns: number[]
}

export type ParticlesResponse = {
  observations: string[]
  diversity: number
}

export type PreviousMoveInfo = {
  player: PlayerColor
  was_observational: boolean
  was_pass: boolean
  captured_stones: number
}

export enum PlayerColor {
  Black = 0,
  White = 1,
}
