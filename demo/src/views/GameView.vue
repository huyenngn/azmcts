<script setup lang="ts">
import { onMounted, ref } from 'vue'
import axios from 'axios'
import GoBoard from '@/components/GoBoard.vue'
import { Button } from '@/components/ui/button'
import {
  PlayerColor,
  type GameStateResponse,
  type MakeMoveRequest,
  type PreviousMoveInfo,
  type StartGameRequest,
  type ParticlesResponse,
} from '@/lib/types'
import { ChevronLeft, RotateCw, Brain } from 'lucide-vue-next'
import MoveInfoHistory from '@/components/MoveInfoHistory.vue'
import {
  Drawer,
  DrawerContent,
  DrawerFooter,
  DrawerTrigger,
  DrawerClose,
} from '@/components/ui/drawer'
import ScrollArea from '@/components/ui/scroll-area/ScrollArea.vue'
import DrawerHeader from '@/components/ui/drawer/DrawerHeader.vue'
import DrawerTitle from '@/components/ui/drawer/DrawerTitle.vue'
import DrawerDescription from '@/components/ui/drawer/DrawerDescription.vue'

const THINKING_DELAY = 300
const NUM_PARTICLES_TO_SHOW = 10

const props = defineProps<{
  playerId: number
  opponentAi: string
}>()

const board = ref<number[]>(Array(81).fill(-1))
const isTerminal = ref<boolean>(false)
const returns = ref<number[]>([0.0, 0.0])
const isLoading = ref<boolean>(false)
const moveHistory = ref<string[]>([])
const particles = ref<string[]>([])
const totalParticles = ref<number>(0)

function parseBoard(observation: string): number[] {
  const rows = observation.matchAll(/\d\s([+OX]+)/g)
  const newBoard: number[] = []
  for (const row of rows) {
    newBoard.push(...Array.from(row[1]).map((cell) => (cell === '+' ? -1 : cell === 'O' ? 1 : 0)))
  }
  return newBoard
}

function formatMoveInfo(info: PreviousMoveInfo): string {
  const player = info.player === PlayerColor.Black ? 'Black' : 'White'
  if (info.was_pass) {
    return `${player} passed.`
  } else if (info.captured_stones > 0) {
    return `${player} captured ${info.captured_stones} ${info.captured_stones > 1 ? 'stones' : 'stone'}.`
  }
  return `${player}'s move was ${info.was_observational ? 'observational' : 'valid'}.`
}

async function startGame() {
  if (isLoading.value) return
  isLoading.value = true
  moveHistory.value = []

  try {
    const response = await axios.post<GameStateResponse>('/start', {
      player_id: props.playerId,
      policy: props.opponentAi,
    } as StartGameRequest)

    if (response.data.observation) {
      board.value = parseBoard(response.data.observation)
    }
    for (const info of response.data.previous_move_infos) {
      moveHistory.value.push(formatMoveInfo(info))
    }
    isTerminal.value = response.data.is_terminal
    returns.value = response.data.returns
  } finally {
    isLoading.value = false
  }
}

async function handleMove(action: number) {
  if (isTerminal.value || isLoading.value) return
  isLoading.value = true

  try {
    const response = await axios.post<GameStateResponse>('/step', {
      action,
    } as MakeMoveRequest)

    if (response.data.observation) {
      board.value = parseBoard(response.data.observation)
    }
    for (const info of response.data.previous_move_infos) {
      moveHistory.value.push(formatMoveInfo(info))
    }
    isTerminal.value = response.data.is_terminal
    returns.value = response.data.returns
  } finally {
    setTimeout(() => {
      isLoading.value = false
    }, THINKING_DELAY)
  }
}

async function fetchParticles() {
  if (isLoading.value) return
  isLoading.value = true
  try {
    const response = await axios.get<ParticlesResponse>(
      `/particles?num_particles=${NUM_PARTICLES_TO_SHOW}`,
    )
    particles.value = response.data.observations
    totalParticles.value = response.data.total
  } catch (error) {
    console.error('Failed to fetch particles:', error)
    particles.value = []
    totalParticles.value = 0
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
          : isLoading
            ? 'AI is thinking...'
            : 'Your turn'
      }}</span>
      <div class="flex items-center gap-2">
        <Button variant="outline" size="icon" @click="startGame" :disabled="isLoading"
          ><RotateCw
        /></Button>
        <Drawer>
          <DrawerTrigger as-child>
            <Button variant="outline" size="icon" @click="fetchParticles" :disabled="isLoading"
              ><Brain
            /></Button>
          </DrawerTrigger>
          <DrawerContent>
            <div class="mx-auto w-full max-w-3xl">
              <DrawerHeader>
                <DrawerTitle>Particle Filter Visualization</DrawerTitle>
                <DrawerDescription
                  >This is a subset of the total {{ totalParticles }} particles sampled from the
                  current belief state.</DrawerDescription
                >
              </DrawerHeader>
              <ScrollArea class="h-96">
                <div class="flex flex-wrap gap-10 justify-between">
                  <GoBoard :board="parseBoard(p)" v-for="(p, index) in particles" :key="index" />
                </div>
              </ScrollArea>
              <DrawerFooter>
                <DrawerClose as-child>
                  <Button variant="outline"> Close </Button>
                </DrawerClose>
              </DrawerFooter>
            </div>
          </DrawerContent>
        </Drawer>
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
