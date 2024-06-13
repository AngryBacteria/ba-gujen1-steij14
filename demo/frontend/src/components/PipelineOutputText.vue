<script setup lang="ts">
import { useUserStore } from '@/stores/user'
import { computed } from 'vue'
import { EntityType, type PipelineEntity } from '@/model/model'

const userStore = useUserStore()

const groupedExtractionResults = computed(() => {
  const grouped = {
    [EntityType.DIAGNOSIS]: [] as PipelineEntity[],
    [EntityType.MEDICATION]: [] as PipelineEntity[],
    [EntityType.TREATMENT]: [] as PipelineEntity[]
  }
  if (!userStore.pipelineData?.entities) return null

  userStore.pipelineData?.entities.forEach((entity) => {
    grouped[entity.entity_type].push(entity)
  })

  return grouped
})

const summary = computed(() => {
  if (!userStore.pipelineData) {
    return null
  } else {
    return userStore.pipelineData.summary
  }
})

function get_attribute_info(entity: PipelineEntity) {
  let filtered = entity.attributes.filter((tag) => tag !== 'POSITIV')
  const output = filtered.map((tag) => {
    if (tag === 'NEGATIV') {
      return { color: 'danger', tag: tag }
    } else if (tag === 'ZUKÜNFTIG' || tag === 'SPEKULATIV') {
      return { color: 'warn', tag: tag }
    } else if (tag === 'LINKS' || tag === 'RECHTS' || tag === 'BEIDSEITIG') {
      return { color: 'info', tag: tag }
    } else {
      return { color: 'success', tag: tag }
    }
  })

  if (entity.normalized_entity) {
    if (entity.entity_type == EntityType.DIAGNOSIS) {
      output.push({ color: 'success', tag: `ICD10GM - ${entity.normalized_entity}` })
    }
    if (entity.entity_type == EntityType.TREATMENT) {
      output.push({ color: 'success', tag: `OPS - ${entity.normalized_entity}` })
    }
    if (entity.entity_type == EntityType.MEDICATION) {
      output.push({ color: 'success', tag: `ATC - ${entity.normalized_entity}` })
    }
  }

  return output
}
</script>

<template>
  <div class="result-layout">
    <Panel header="Zusammenfassung" toggleable v-if="summary">
      <p>{{ summary }}</p>
    </Panel>

    <Panel
      header="Diagnosen"
      toggleable
      v-if="groupedExtractionResults && groupedExtractionResults[EntityType.DIAGNOSIS].length > 0"
    >
      <section v-for="(entity, index) in groupedExtractionResults[EntityType.DIAGNOSIS]">
        <p v-if="entity?.normalized_entity">
          {{ entity.entity }}
        </p>
        <p v-else>{{ entity.entity }}</p>
        <span v-for="attribute in get_attribute_info(entity)" class="attribute-tag">
          <Tag :severity="attribute.color" :value="attribute.tag"
        /></span>
        <Divider v-if="index + 1 !== groupedExtractionResults[EntityType.DIAGNOSIS].length" />
      </section>
    </Panel>

    <Panel
      header="Medikamente"
      toggleable
      v-if="groupedExtractionResults && groupedExtractionResults[EntityType.MEDICATION].length > 0"
    >
      <section v-for="(entity, index) in groupedExtractionResults[EntityType.MEDICATION]">
        <p v-if="entity?.normalized_entity">
          {{ entity.entity }}
        </p>
        <p v-else>{{ entity.entity }}</p>
        <span v-for="attribute in get_attribute_info(entity)" class="attribute-tag">
          <Tag :severity="attribute.color" :value="attribute.tag"
        /></span>
        <Divider v-if="index + 1 !== groupedExtractionResults[EntityType.MEDICATION].length" />
      </section>
    </Panel>

    <Panel
      header="Prozeduren"
      toggleable
      v-if="groupedExtractionResults && groupedExtractionResults[EntityType.TREATMENT].length > 0"
    >
      <section v-for="(entity, index) in groupedExtractionResults[EntityType.TREATMENT]">
        <p v-if="entity?.normalized_entity">
          {{ entity.entity }}
        </p>
        <p v-else>{{ entity.entity }}</p>
        <span v-for="attribute in get_attribute_info(entity)" class="attribute-tag">
          <Tag :severity="attribute.color" :value="attribute.tag"
        /></span>
        <Divider v-if="index + 1 !== groupedExtractionResults[EntityType.TREATMENT].length" />
      </section>
    </Panel>
  </div>
</template>

<style scoped>
.result-layout {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.attribute-tag {
  margin-right: 0.5rem;
  margin-bottom: 0.5rem;
}
</style>
