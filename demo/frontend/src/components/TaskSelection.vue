<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useUserStore } from '@/stores/user'
import { storeToRefs } from 'pinia'

const userStore = useUserStore()
const { tasks } = storeToRefs(userStore)

// Computed and watcher Make sure that the right state for the normalization task is set
const normalizationAvailable = computed(() => {
  return tasks.value.includes('extraction')
})
watch(tasks, (newTasks, oldTasks) => {
  if (!newTasks.includes('extraction') && oldTasks.includes('extraction')) {
    tasks.value = tasks.value.filter((task) => task !== 'normalization')
  }
})

// Modes
const analysisModes = ref([
  {
    name: 'Alle Entitäten mit Attributen',
    value: 'bronco'
  },
  {
    name: 'Nur Medikation mit erweiterten Attributen',
    value: 'cardio'
  }
])
</script>

<template>
  <div class="checkbox-group">
    <div class="centered-flexbox">
      <Checkbox
        v-model="tasks"
        inputId="extraction"
        name="extraction"
        value="extraction"
        :disabled="userStore.ongoingRequest"
      />
      <label for="extraction"> Extrahieren </label>
    </div>
    <div class="centered-flexbox">
      <Checkbox
        v-model="tasks"
        inputId="normalization"
        name="normalization"
        value="normalization"
        :disabled="!normalizationAvailable || userStore.ongoingRequest"
      />
      <label for="normalization"> Normalisieren </label>
    </div>
    <div class="centered-flexbox">
      <Checkbox
        v-model="tasks"
        inputId="summary"
        name="summary"
        value="summary"
        :disabled="userStore.ongoingRequest"
      />
      <label for="summary"> Zusammenfassen </label>
    </div>

    <Select
      v-model="userStore.attributeFormat"
      :options="analysisModes"
      optionLabel="name"
      optionValue="value"
      placeholder="Modus auswählen"
      :loading="userStore.ongoingRequest"
      :disabled="userStore.ongoingRequest"
    />
  </div>
</template>

<style scoped>
.checkbox-group {
  display: flex;
  flex-direction: row;
  gap: 2rem;
}
.centered-flexbox {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 0.5rem;
}
</style>
