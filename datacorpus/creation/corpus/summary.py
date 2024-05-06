import os
import random

from dotenv import load_dotenv
from openai import OpenAI

from datacorpus.utils.mongodb import upload_data_to_mongodb

# Creation of summaries for various sources which can be used for training the model for the task of summarizing text.

LC2_DATA_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\divers\\lc2"
CLEF_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\corpus\\clef ehealth\\docs-training"

SUMMARIZATION_PROMPT = """Bitte fasse den folgenden medizinischen Text präzise zusammen. Gib mir nur die Zusammenfassung 
und keinen weiteren Text. Achte darauf, alle wesentlichen medizinischen Informationen und 
Schlüsseldaten beizubehalten:\n\n<<CONTEXT>>\n"""

load_dotenv()
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    # This is the default and can be omitted
    api_key=openai_key,
)


def create_summary_data_from_lc2():
    """
    Create summary data from the lc2 data used in the last semester. Consists of 26 files with synthetic
    medical dialoges.
    """
    output = []
    model = "gpt-3.5-turbo"

    files = [f for f in os.listdir(LC2_DATA_PATH) if f.endswith(".txt")]
    for file in files:
        with open(os.path.join(LC2_DATA_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            summary = get_summary_of_text(text, model)
            output.append(
                {
                    "origin": text,
                    "summary": summary,
                    "source": "lc2",
                    "model": model,
                }
            )

            return output


def create_summary_data_from_clef2019(amount=50):
    """
    Create summaries for clef 2019 data.
    """
    output = []
    model = "gpt-3.5-turbo"
    files = [f for f in os.listdir(CLEF_PATH) if f.endswith(".txt")]
    files = random.sample(files, amount)

    for file in files:
        with open(os.path.join(CLEF_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            summary = get_summary_of_text(text, model)
            output.append(
                {
                    "origin": text,
                    "summary": summary,
                    "source": "clef",
                    "model": model,
                }
            )

    return output


def get_summary_of_text(text: str, model: str):
    """
    Get a summary of a given text using the OpenAI API.
    """
    text = text.strip()
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": SUMMARIZATION_PROMPT.replace("<<CONTEXT>>", text),
            },
        ],
        model=model,
    )
    summary = response.choices[0].message.content
    summary = summary.strip()
    return summary


def upload_summary_data_to_mongodb():
    """
    Upload the summary data to the mongodb.
    """
    data_lc2 = create_summary_data_from_lc2()
    data_clef = create_summary_data_from_clef2019(50)

    data = data_lc2 + data_clef
    upload_data_to_mongodb(data, "corpus", "summarization", True, [])


upload_summary_data_to_mongodb()
