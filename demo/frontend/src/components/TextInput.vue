<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useUserStore } from '@/stores/user'
import { storeToRefs } from 'pinia'

const userStore = useUserStore()
const { tasks, ongoingRequest } = storeToRefs(userStore)

const noTask = computed(() => tasks.value.length === 0)

const inputText = ref('')

const selectedExample = ref()
const examples = ref([
  {
    name: 'Patientin mit Brustschmerz',
    text: 'Die Patientin Elizabeth Brönnimann hatte sich heute in der Notfallaufnahme gemeldet weil sie Brustschmerzen hatte.'
  }
])

watch(
  () => selectedExample.value,
  (newVal) => {
    inputText.value = newVal.text
  }
)
</script>

<template>
  <section class="input-layout">
    <Textarea v-model="inputText" rows="5" class="text-area" :disabled="ongoingRequest" />

    <section class="text-actions">
      <div>
        <Button
          v-if="noTask"
          label="Wählen sie mindestens eine Aufhabe aus"
          icon="pi pi-ban"
          size="large"
          class="analyze-button"
          :disabled="noTask || ongoingRequest"
        />
        <Button
          v-else
          @click="userStore.getAnalysis(inputText)"
          label="Text Analysieren"
          icon="pi pi-microchip-ai"
          size="large"
          class="analyze-button"
          :loading="ongoingRequest"
        />
      </div>

      <Select
        v-model="selectedExample"
        :options="examples"
        optionLabel="name"
        placeholder="Beispiel auswählen"
        :disabled="ongoingRequest"
      />
    </section>
  </section>
</template>

<style scoped>
.text-area {
  width: 100%;
}

.input-layout {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.text-actions {
  display: flex;
  flex-direction: row;
  gap: 1rem;
}
</style>
