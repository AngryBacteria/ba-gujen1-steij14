export interface PipelineBody {
  text: string
  extraction: boolean
  normalization: boolean
  summary: boolean
  entity_types?: EntityType[]
  attribute_format?: AttributeFormat
}

export interface PipelineResponse {
  entities: PipelineEntity[]
  summary: string
  execution_time: number
}

export interface PipelineEntity {
  entity_type: EntityType
  origin: string
  entity: string
  attributes: string[]
  raw_output: string
  normalized_entity: string
}

export enum EntityType {
  DIAGNOSIS = 'DIAGNOSIS',
  TREATMENT = 'TREATMENT',
  MEDICATION = 'MEDICATION'
}

export enum AttributeFormat {
  BRONCO = 'bronco',
  CARDIO = 'cardio'
}
