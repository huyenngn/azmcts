<script setup lang="ts">
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
import { Button } from '@/components/ui/button'
import GoBoard from '@/components/GoBoard.vue'
import { Brain } from 'lucide-vue-next'

const props = defineProps<{
  particleDiversity: number
  particles: number[][]
  disabled: boolean
}>()

const emit = defineEmits<{
  (e: 'fetchParticles'): void
}>()
</script>

<template>
  <Drawer>
    <DrawerTrigger as-child>
      <Button
        variant="outline"
        size="icon"
        @click="emit('fetchParticles')"
        :disabled="props.disabled"
        ><Brain
      /></Button>
    </DrawerTrigger>
    <DrawerContent>
      <div class="mx-auto w-full max-w-3xl">
        <DrawerHeader>
          <DrawerTitle>Particle Filter Visualization</DrawerTitle>
          <DrawerDescription
            >This is a subset of the current belief state. Particle diversity is at
            {{ (props.particleDiversity * 100).toFixed(2) }}%.
          </DrawerDescription>
        </DrawerHeader>
        <ScrollArea class="h-96">
          <div class="flex flex-wrap gap-10 justify-between pointer-events-none">
            <GoBoard :board="p" v-for="(p, index) in props.particles" :key="index" />
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
</template>
