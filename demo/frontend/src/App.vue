<script setup lang="ts">
import { RouterView } from 'vue-router'
</script>

<template>
  <div class="page-wrapper">
    <RouterView />
  </div>
</template>

<style scoped>
.checkbox-group {
  display: flex;
  flex-direction: row;
  gap: 1rem;
}
.centered-flexbox {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 0.5rem;
}
.page-wrapper {
  margin-right: 5%;
  margin-left: 5%;
}
</style>
