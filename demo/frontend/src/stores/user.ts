import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import type { PipelineBody, PipelineResponse } from '@/model/model'

export const useUserStore = defineStore('user', () => {
  // task configs
  const tasks = ref(['extraction', 'normalization'])
  const inputText = ref('')

  // Request data
  const ongoingRequest = ref(false)
  const pipelineData = ref<PipelineResponse | null>(null)

  async function getAnalysis() {
    try {
      ongoingRequest.value = true
      pipelineData.value = null

      const reqBody: PipelineBody = {
        text: inputText.value,
        extraction: tasks.value.includes('extraction'),
        normalization: tasks.value.includes('normalization'),
        summary: tasks.value.includes('summary')
      }
      const response = await fetch('http://127.0.0.1:8000/pipeline', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(reqBody)
      })

      const data = await response.json()
      if (response.ok) {
        console.log(data)
        pipelineData.value = data
      }
      return data
    } catch (error) {
      console.error(error)
    } finally {
      ongoingRequest.value = false
    }
  }

  return {
    tasks,
    ongoingRequest,
    getAnalysis,
    pipelineData,
    inputText
  }
})
