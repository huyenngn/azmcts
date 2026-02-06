export type StartGameRequest = {
  player_id: number
  policy: string
}

export type MakeMoveRequest = {
  action: number
}

export type GameStateResponse = {
  observation: string
  previous_move_infos: PreviousMoveInfo[]
  is_terminal: boolean
  returns: number[]
}

export type ParticlesResponse = {
  observations: string[]
  total: number
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
