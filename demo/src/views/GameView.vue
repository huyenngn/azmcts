<script setup lang="ts">
import { onMounted, ref } from 'vue'
import GoBoard from '@/components/GoBoard.vue'
import { Button } from '@/components/ui/button'
import {
  type MakeMoveRequest,
  type StartGameRequest,
  type ParticlesResponse,
  type GameStateResponse,
} from '@/lib/types'
import { ChevronLeft, RotateCw } from 'lucide-vue-next'
import MoveInfoHistory from '@/components/MoveInfoHistory.vue'
import ParticlesVisualizer from '@/components/ParticlesVisualizer.vue'
import { getBackend, streamGameUpdates } from '@/lib/requests'
import { formatMoveInfo, parseBoard } from '@/lib/game'

const NUM_PARTICLES_TO_SHOW = 10

const props = defineProps<{
  playerId: number
  opponentAi: string
}>()

const board = ref<number[]>(Array(81).fill(-1))
const isTerminal = ref<boolean>(false)
const returns = ref<number[]>([0.0, 0.0])
const isLoading = ref<boolean>(false)
const isAiThinking = ref<boolean>(false)
const moveHistory = ref<string[]>([])
const particles = ref<number[][]>([])
const particleDiversity = ref<number>(0)

function updateGameState(data: GameStateResponse) {
  if (data.current_player !== props.playerId) {
    isAiThinking.value = true
  } else {
    isAiThinking.value = false
  }
  if (data.observation) {
    board.value = parseBoard(data.observation)
  }
  if (data.previous_move_info) {
    moveHistory.value.push(formatMoveInfo(data.previous_move_info))
  }
  isTerminal.value = data.is_terminal
  returns.value = data.returns
}

async function startGame() {
  if (isLoading.value) return
  isLoading.value = true
  moveHistory.value = []

  try {
    await streamGameUpdates(
      '/start',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          player_id: props.playerId,
          policy: props.opponentAi,
        } as StartGameRequest),
      },
      {
        onUpdate: (data) => {
          if (data.current_player !== props.playerId) {
            isAiThinking.value = true
          } else {
            isAiThinking.value = false
          }
          if (data.observation) {
            board.value = parseBoard(data.observation)
          }
          if (data.previous_move_info) {
            moveHistory.value.push(formatMoveInfo(data.previous_move_info))
          }
          isTerminal.value = data.is_terminal
          returns.value = data.returns
        },
        onError: (data) => {
          console.error('Streaming error:', data)
        },
      },
    )
  } finally {
    isLoading.value = false
    isAiThinking.value = false
  }
}

async function handleMove(action: number) {
  if (isTerminal.value || isLoading.value) return
  isLoading.value = true

  try {
    await streamGameUpdates(
      '/step',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action,
        } as MakeMoveRequest),
      },
      {
        onUpdate: (data: GameStateResponse) => {
          updateGameState(data)
        },
        onError: (data) => {
          console.error('Streaming error:', data)
        },
      },
    )
  } finally {
    isLoading.value = false
    isAiThinking.value = false
  }
}

async function fetchParticles() {
  if (isLoading.value) return
  isLoading.value = true

  try {
    const data = await getBackend<ParticlesResponse>(
      `/particles?num_particles=${NUM_PARTICLES_TO_SHOW}`,
    )

    particles.value = []
    for (const observation of data.observations) {
      const particle = parseBoard(observation)
      particles.value.push(particle)
    }
    particleDiversity.value = data.diversity
  } catch (error) {
    console.error('Failed to fetch particles:', error)
    particles.value = []
    particleDiversity.value = 0
  } finally {
    isLoading.value = false
  }
}

onMounted(() => {
  startGame()
})
</script>

<template>
  <div class="grow flex items-stretch justify-center flex-col gap-4">
    <div class="flex items-center justify-between gap-4">
      <RouterLink to="/">
        <Button variant="outline" size="icon" :disabled="isLoading"><ChevronLeft /></Button>
      </RouterLink>
      <span>{{
        isTerminal
          ? returns[props.playerId] > returns[1 - props.playerId]
            ? 'You win!'
            : returns[props.playerId] < returns[1 - props.playerId]
              ? 'You lose!'
              : 'Draw'
          : isAiThinking
            ? 'AI is thinking...'
            : 'Your turn'
      }}</span>
      <div class="flex items-center gap-2">
        <Button variant="outline" size="icon" @click="startGame" :disabled="isLoading"
          ><RotateCw
        /></Button>
        <ParticlesVisualizer
          :particleDiversity="particleDiversity"
          :particles="particles"
          :disabled="isLoading"
          @fetchParticles="fetchParticles"
        />
      </div>
    </div>
    <div
      :class="{ 'opacity-50 pointer-events-none': isLoading || isTerminal }"
      class="grow sm:grow-0 flex items-stretch sm:items-stretch justify-stretch sm:justify-between gap-8 sm:gap-0 sm:flex-row flex-col"
    >
      <GoBoard :board="board" @move="handleMove" />
      <div class="grow flex flex-col items-stretch justify-between gap-4">
        <Button @click="handleMove(81)">Pass</Button>
        <MoveInfoHistory :previousMoveInfos="moveHistory" />
      </div>
    </div>
  </div>
</template>
