from fastapi import FastAPI

from shared.decoder_utils import (
    load_model_and_tokenizer,
    generate_output,
    get_extractions_with_attributes_grouped,
    get_model_output_only,
)
from shared.prompt_utils import (
    get_extraction_messages,
    AttributeFormat,
    EntityType,
    get_normalization_messages,
)

app = FastAPI()
tokenizer, model = load_model_and_tokenizer()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/pipeline")
def execute_pipeline(
    text: str,
    normalize: bool,
    entity_types: list[EntityType] = None,
    attribute_format=AttributeFormat.BRONCO,
):
    if entity_types is None:
        entity_types = [
            EntityType.DIAGNOSIS,
            EntityType.TREATMENT,
            EntityType.MEDICATION,
        ]

    entities = []
    for entity_type in entity_types:
        try:
            messages = get_extraction_messages(text, attribute_format, entity_type)
            raw_output, _ = generate_output(messages, model, tokenizer)
            output = get_model_output_only(raw_output)
            output_parsed = get_extractions_with_attributes_grouped(output)

            for key, value in output_parsed.items():
                if key.lower() != "keine vorhanden":
                    entities.append(
                        {
                            "entity_type": entity_type,
                            "origin": text,
                            "entity": key,
                            "attributes": value,
                        }
                    )
        except Exception as e:
            print(e)

    if normalize:
        for entity in entities:
            try:
                messages = get_normalization_messages(
                    entity["entity"], entity["origin"], entity["entity_type"]
                )
                raw_output, _ = generate_output(messages, model, tokenizer)
                output = get_model_output_only(raw_output)
                entity["normalized_entity"] = output
            except Exception as e:
                print(e)

    return entities
