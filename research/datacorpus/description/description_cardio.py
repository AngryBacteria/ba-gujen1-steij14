# TODO: properties graphs and tables of cardio
import pandas as pd
from pandas import DataFrame

from research.datacorpus.creation.utils.utils_mongodb import get_collection

def save_to_csv() -> None:
    """
    Save the cardio mongodb to trimmed down csv file
    :return: None
    """

    cardio_collection = get_collection("corpus", "cardio")
    cardio_cursor = cardio_collection.find({})

    print(cardio_collection.count_documents({}))

    formatted_annotations = []
    for doc in cardio_cursor:
        for anno in doc["annotations"]:
            if anno["type"] == "NA":
                formatted_annotations.append(
                    {"type": anno["type"], "origin": anno["origin"], "text": "NA"}
                )
            else:
                for text in anno["text"]:
                    formatted_annotations.append(
                        {"type": anno["type"], "origin": anno["origin"], "text": text}
                    )

    df = pd.DataFrame(formatted_annotations)
    df.to_csv("cardio_description.csv", sep="|", index=False)

def read_from_csv() -> DataFrame:
    """
    Read the cardio properties csv file
    :return: cardio properties dataframe
    """
    df = pd.read_csv("cardio_description.csv", sep="|", na_filter=False)
    return df

def type_pieplot(df: DataFrame) -> None:
    """
    Plot the distribution of the extraction types
    :param df: The dataframe to plot from
    :return: None
    """
    import plotly.express as px

    type_counts = df["type"].value_counts()
    type_counts_df = pd.DataFrame(
        {"type": type_counts.index, "count": type_counts.values}
    )
    fig = px.pie(
        type_counts_df, values="count", names="type", title="Distribution of Types in Cardio Corpus"
    )
    fig.show()


def paragraph_lengths(df, tokenize=False) -> tuple:
    """
    Get the max, min, average and median length of the paragraphs
    :param df: The dataframe to calculate from. Needs to have an "origin" column
    :param tokenize: If the lengths should be calculated by tokens or not
    :return: The max, min, average and median length of the paragraphs
    """
    lengths = []
    if tokenize:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "LeoLM/leo-mistral-hessianai-7b", use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for index, row in df.iterrows():
            paragraph_length = len(tokenizer.tokenize(row["origin"]))
            lengths.append(paragraph_length)
    else:
        for index, row in df.iterrows():
            paragraph_length = len(row["origin"])
            lengths.append(paragraph_length)
    # max
    max_length = max(lengths)
    # min
    min_length = min(lengths)
    # average
    avg_length = sum(lengths) / len(lengths)
    # median
    lengths.sort()
    if len(lengths) % 2 == 0:
        median = (lengths[len(lengths) // 2 - 1] + lengths[len(lengths) // 2]) / 2
    else:
        median = lengths[len(lengths) // 2]

    print(
        f"Max: {max_length}, Min: {min_length}, Average: {avg_length}, Median: {median}"
    )
    return max_length, min_length, avg_length, median

def plot_lengths_boxplot(df: DataFrame, tokenize=False) -> None:
    """
    Plot the distribution of paragraph lengths
    :param df: The dataframe to plot from
    :param tokenize: Whether to calculate lengths based on tokenization
    :return: None
    """
    import plotly.express as px

    # calculate the lengths
    lengths = []
    if tokenize:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "LeoLM/leo-mistral-hessianai-7b", use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for index, row in df.iterrows():
            paragraph_length = len(tokenizer.tokenize(row["origin"]))
            lengths.append(paragraph_length)
    else:
        for index, row in df.iterrows():
            paragraph_length = len(row["origin"])
            lengths.append(paragraph_length)

    # display the boxplot
    fig = px.box(y=lengths, title="Paragraph Lengths in Cardio Corpus")
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Length",
        showlegend=False
    )
    fig.show()

save_to_csv()
df_main = read_from_csv()
type_pieplot(df_main)
paragraph_lengths(df_main, tokenize=True)
plot_lengths_boxplot(df_main)
