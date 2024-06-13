<script setup lang="ts">
import TaskSelection from '@/components/TaskSelection.vue'
import TextInput from '@/components/TextInput.vue'
import PipelineOutputText from '@/components/PipelineOutputText.vue'
</script>

<template>
  <main>
    <h1>Medizinische Textanalyse mit Gemma-2b</h1>

    <section class="main-layout">
      <TaskSelection />
      <TextInput />
      <PipelineOutputText />
    </section>
  </main>
</template>

<style scoped>
.main-layout {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
</style>
