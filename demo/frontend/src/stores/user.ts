import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useUserStore = defineStore('user', () => {
  const tasks = ref(['extraction', 'normalization'])
  const ongoingRequest = ref(false)

  async function getAnalysis(origin: string) {
    try {
      ongoingRequest.value = true
      const response = await fetch('http://127.0.0.1:8000/pipeline', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: origin,
          extraction: tasks.value.includes('extraction'),
          normalization: tasks.value.includes('normalization'),
          summary: tasks.value.includes('summary'),
          entity_types: ['DIAGNOSIS', 'TREATMENT', 'MEDICATION'],
          attribute_format: 'bronco'
        })
      })

      const data = await response.json()
      console.log(data)

      return data
    } finally {
      ongoingRequest.value = false
    }
  }

  return {
    tasks,
    ongoingRequest,
    getAnalysis
  }
})
