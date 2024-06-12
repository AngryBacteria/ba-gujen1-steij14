<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useUserStore } from '@/stores/user'
import { storeToRefs } from 'pinia'

const userStore = useUserStore()
const { tasks, inputText } = storeToRefs(userStore)

const noTask = computed(() => tasks.value.length === 0)

const selectedExample = ref()
const examples = ref([
  {
    name: 'Patientin mit Brustschmerz',
    text: 'Die Patientin Elizabeth Brönnimann hatte sich heute in der Notfallaufnahme gemeldet weil sie Brustschmerzen hatte. Es wurde ebenfalls ein Röntgenbild erstellt, dabei zeigte sich aber keine Lungenentzündung.'
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
    <Textarea v-model="inputText" rows="5" class="text-area" :disabled="userStore.ongoingRequest" />

    <section class="text-actions">
      <div>
        <Button
          v-if="noTask"
          label="Wählen sie mindestens eine Aufhabe aus"
          icon="pi pi-ban"
          size="large"
          class="analyze-button"
          :disabled="noTask || userStore.ongoingRequest"
        />
        <Button
          v-else-if="inputText.length === 0"
          label="Für die Analyse ist Text benötigt"
          icon="pi pi-ban"
          size="large"
          class="analyze-button"
          :disabled="true"
        />
        <Button
          v-else
          @click="userStore.getAnalysis"
          label="Text Analysieren"
          icon="pi pi-microchip-ai"
          size="large"
          class="analyze-button"
          :loading="userStore.ongoingRequest"
        />
      </div>

      <Select
        v-model="selectedExample"
        :options="examples"
        optionLabel="name"
        placeholder="Beispiel auswählen"
        :disabled="userStore.ongoingRequest"
      />

      <p v-if="userStore.pipelineData && userStore.pipelineData?.execution_time">
        Die Analyse hat {{ userStore.pipelineData.execution_time }} Millisekunden gedauert
      </p>
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
