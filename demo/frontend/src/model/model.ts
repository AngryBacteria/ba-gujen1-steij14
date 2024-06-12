export interface AnalysisBody {
  text: string
  extraction: boolean
  normalization: boolean
  summary: boolean
  entity_types: string[]
  attribute_format: string
}

export interface PipelineEntityResponse {
  entities: PipelineEntity[]
  summary: string
}

export interface PipelineEntity {
  entity_type: string
  origin: string
  entity: string
  attributes: string[]
  raw_output: string
  normalized_entity: string
}
