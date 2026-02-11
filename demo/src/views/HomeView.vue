<script setup lang="ts">
import { Button } from '@/components/ui/button'
import { ref } from 'vue'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { PlayerColor } from '@/lib/types'

const playerId = ref(PlayerColor.Black)
const opponentAi = ref('azbsmcts')
</script>

<template>
  <main class="grow flex items-center justify-center">
    <div class="flex flex-col items-center gap-4">
      <h1 class="font-black">azbsmcts</h1>
      <table class="table-auto [&_td]:p-2">
        <tbody>
          <tr>
            <td>Choose Player:</td>
            <td>
              <Select v-model="playerId">
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem :value="PlayerColor.White"> White </SelectItem>
                    <SelectItem :value="PlayerColor.Black"> Black </SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </td>
          </tr>
          <tr>
            <td>Choose Opponent:</td>
            <td>
              <Select v-model="opponentAi">
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="random"> Randomizer </SelectItem>
                    <SelectItem value="bsmcts"> BSMCTS </SelectItem>
                    <SelectItem value="azbsmcts"> AZ+BSMCTS </SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
            </td>
          </tr>
        </tbody>
      </table>
      <RouterLink :to="`/game?playerId=${playerId}&opponentAi=${opponentAi}`">
        <Button>Start Game</Button>
      </RouterLink>
    </div>
  </main>
</template>

<style scoped>
td:nth-child(even) {
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
