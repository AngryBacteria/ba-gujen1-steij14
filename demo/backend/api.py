from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from demo.backend.singletons import Cache, ModelTokenizer
from shared.decoder_utils import (
    generate_output,
    get_extractions_with_attributes_grouped,
    get_model_output_only,
)
from shared.logger import logger
from shared.prompt_utils import (
    get_extraction_messages,
    AttributeFormat,
    EntityType,
    get_normalization_messages,
    get_summarization_messages,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app_cache = Cache(max_items=100)
app_models = ModelTokenizer()


@app.get("/")
def read_root():
    return {"Hello": "LLM"}


@app.get("/timeout")
def read_timeout():
    import time

    time.sleep(2)
    return {"Hello": "LLM"}


class PipelineConfig(BaseModel):
    text: str  # the text to process
    extraction: bool  # whether to perform extraction or not
    normalization: bool  # whether to perform normalization or not
    summary: bool  # whether to perform summarization or not
    entity_types: Optional[List[EntityType]] = [
        EntityType.DIAGNOSIS,
        EntityType.TREATMENT,
        EntityType.MEDICATION,
    ]
    attribute_format: AttributeFormat = AttributeFormat.BRONCO


class PipelineEntity(BaseModel):
    entity_type: EntityType
    origin: str
    entity: str
    attributes: list[str]
    raw_output: str
    normalized_entity: Optional[str] = None


class PipelineResponse(BaseModel):
    entities: List[PipelineEntity]
    summary: str


@app.post("/pipeline")
def execute_pipeline(config: PipelineConfig) -> PipelineResponse:
    if config.extraction is False and config.normalization is False:
        raise HTTPException(
            status_code=400, detail="Mindestens eine Aufgabe muss aktiviert sein."
        )

    # check cache
    cache_key = f"{config.text}_{config.extraction}_{config.normalization}_{config.summary}_{config.entity_types}_{config.attribute_format}"
    cached_result = app_cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"Returning cached result for key: {cache_key}")
        return cached_result

    entities = []
    for entity_type in config.entity_types:
        try:
            messages = get_extraction_messages(
                config.text, config.attribute_format, entity_type
            )
            raw_output, _ = generate_output(
                messages, app_models.model, app_models.tokenizer
            )
            output = get_model_output_only(raw_output)
            output_parsed = get_extractions_with_attributes_grouped(output)

            for key, value in output_parsed.items():
                if key.lower() != "keine vorhanden":
                    entities.append(
                        PipelineEntity(
                            entity_type=entity_type,
                            origin=config.text,
                            entity=key,
                            attributes=value,
                            raw_output=output,
                        )
                    )
        except Exception as e:
            print(e)

    if config.normalization:
        for entity in entities:
            try:
                messages = get_normalization_messages(
                    entity.entity, entity.origin, entity.entity_type
                )
                raw_output, _ = generate_output(
                    messages, app_models.model, app_models.tokenizer
                )
                output = get_model_output_only(raw_output)
                entity.normalized_entity = output
            except Exception as e:
                print(e)

    summary = ""
    if config.summary:
        messages = get_summarization_messages(config.text)
        raw_output, _ = generate_output(
            messages, app_models.model, app_models.tokenizer
        )
        output = get_model_output_only(raw_output)
        summary = output

    # update cache
    app_cache.set(cache_key, PipelineResponse(entities=entities, summary=summary))

    logger.debug(f"Returning generated result for key: {cache_key}")
    return PipelineResponse(entities=entities, summary=summary)
