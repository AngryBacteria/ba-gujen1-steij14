<script setup lang="ts">
import TaskSelection from '@/components/TaskSelection.vue'
import TextInput from '@/components/TextInput.vue'
</script>

<template>
  <main>
    <h1>Bachelorarbeit Textanalyse mit Gemma-2b</h1>

    <section class="main-layout">
      <TaskSelection />
      <TextInput />
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
