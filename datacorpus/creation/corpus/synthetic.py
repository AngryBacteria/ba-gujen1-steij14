import datetime
import os
import random

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from shared.logger import logger
from shared.model_utils import count_tokens
from shared.mongodb import upload_data_to_mongodb, get_collection

# Creation of data with synthetic methods such as the OpenAI API.

LC2_DATA_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\synthetic\\lc2_dialogs"
CLEF_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus\\clef ehealth\\docs-training"

SUMMARISE_TEXT_PROMPT = """Bitte fasse den folgenden medizinischen Text präzise zusammen. Gib mir nur die Zusammenfassung 
und keinen weiteren Text. Achte darauf, alle wesentlichen medizinischen Informationen und 
Schlüsseldaten beizubehalten:\n\n<<CONTEXT>>\n"""

load_dotenv()
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    api_key=openai_key,
)


def create_summary_data_from_lc2():
    """
    Create summary data from the lc2 data used in the last semester. Consists of 26 files with synthetic
    medical dialogs.
    """
    output = []
    model = "gpt-4-turbo"

    files = [f for f in os.listdir(LC2_DATA_PATH) if f.endswith(".txt")]
    for file in files:
        with open(os.path.join(LC2_DATA_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            summary, usage, openai_model, execution_time = get_summary_of_text(
                text, model
            )
            output.append(
                {
                    "origin": text,
                    "summary": summary,
                    "source": "lc2",
                    "model": openai_model,
                    "token_count": usage.total_tokens,
                    "execution_time": execution_time,
                    "task": "summarization",
                }
            )

    logger.debug(f"Created {len(output)} summaries from lc2 data.")
    return output


def create_summary_data_from_clef2019(amount: int):
    """
    Create summaries for clef 2019 data.
    :param amount: Amount of summaries to create.
    """
    output = []
    model = "gpt-4-turbo"
    files = [f for f in os.listdir(CLEF_PATH) if f.endswith(".txt")]
    files = random.sample(files, amount)

    for file in files:
        with open(os.path.join(CLEF_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            summary, usage, openai_model, execution_time = get_summary_of_text(
                text, model
            )
            output.append(
                {
                    "origin": text,
                    "summary": summary,
                    "source": "clef",
                    "model": openai_model,
                    "token_count": usage.total_tokens,
                    "execution_time": execution_time,
                    "task": "summarization",
                }
            )

    logger.debug(f"Created {len(output)} summaries from clef data.")
    return output


def get_dev4med_data():
    """
    Get the data from the Dev4Med dataset.
    """
    dataset = load_dataset("Dev4Med/Notfallberichte-German-100")
    output = []
    for data in dataset["train"]:
        output.append(
            {
                "origin": data["source"].strip(),
                "summary": data["summary"].strip(),
                "source": "Dev4Med/Notfallberichte-German-100",
                "task": "summarization",
            }
        )
    logger.debug(f"Created {len(output)} summaries from Dev4Med data.")
    return output


def get_summary_of_text(text: str, model: str):
    """
    Get a summary of a given text using the OpenAI API.
    """
    text = text.strip()
    _start_time = datetime.datetime.now()
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": SUMMARISE_TEXT_PROMPT.replace("<<CONTEXT>>", text),
            },
        ],
        model=model,
    )
    summary = response.choices[0].message.content
    summary = summary.strip()

    _end_time = datetime.datetime.now()
    execution_time = (_end_time - _start_time).microseconds

    logger.debug(
        f""
        f"Created summary in {execution_time} microseconds,"
        f"Tokens: {response.usage.total_tokens},"
        f"Model: {model}"
    )
    return summary, response.usage, response.model, execution_time


def count_synthetic_tokens():
    """
    Count the tokens of the synthetic data. Separated into origin, summary and total count
    """
    texts_origin = []
    texts_summary = []
    synthetic_collection = get_collection("corpus", "synthetic")
    documents = synthetic_collection.find({})
    for doc in documents:
        texts_origin.append(doc["origin"])
        texts_summary.append(doc["summary"])

    tokens_origin = count_tokens(texts_origin, None, "LeoLM/leo-mistral-hessianai-7b")
    tokens_summary = count_tokens(texts_summary, None, "LeoLM/leo-mistral-hessianai-7b")
    tokens = tokens_summary + tokens_origin

    return tokens_origin, tokens_summary, tokens


def upload_summary_data_to_mongodb(lc2: bool, clef: bool, dev4med: bool):
    """
    Upload the summary data to the mongodb.
    """
    data = []
    if lc2:
        data_lc2 = create_summary_data_from_lc2()
        data.extend(data_lc2)
    if clef:
        data_clef = create_summary_data_from_clef2019(100)
        data.extend(data_clef)
    if dev4med:
        data_dev4med = get_dev4med_data()
        data.extend(data_dev4med)

    upload_data_to_mongodb(data, "corpus", "synthetic", True, [])


# print(count_synthetic_tokens())
upload_summary_data_to_mongodb(True, True, True)
