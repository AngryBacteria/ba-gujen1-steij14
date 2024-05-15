import pandas as pd
from pandas import DataFrame

from shared.model_utils import patch_tokenizer_with_template
from shared.mongodb import get_collection


# Collection of functions to analyze the synthetic dataset.


def save_to_csv():
    """
    Saves the synthetic collection to a trimmed down csv file. This way the database can be queried only once.
    """
    synthetic_collection = get_collection("corpus", "synthetic")
    synthetic_cursor = synthetic_collection.find({})
    output = []
    for doc in synthetic_cursor:
        output.append(doc)

    df = pd.DataFrame(output)
    df.to_csv("synthetic_description.csv", index=False, sep="|")


def read_from_csv() -> DataFrame:
    """
    Reads the collection data.
    """
    df = pd.read_csv("synthetic_description.csv", sep="|", na_filter=False)
    return df


def show_text_length_boxplot(df: DataFrame, text_type: str, tokenize: bool = False):
    """
    Boxplot showing the length of the text.
    :param df: DataFrame the data
    :param text_type: the type of text to count (origin or summary)
    :param tokenize: whether to tokenize the text or count by characters
    """
    import plotly.express as px

    # calculate the lengths
    lengths = []
    if tokenize:
        tokenizer = patch_tokenizer_with_template()

        for index, row in df.iterrows():
            paragraph_length = len(tokenizer.tokenize(row[text_type]))
            lengths.append(paragraph_length)
    else:
        for index, row in df.iterrows():
            paragraph_length = len(row[text_type])
            lengths.append(paragraph_length)

    # display the boxplot
    fig = px.box(y=lengths)
    fig.update_layout(
        title=f"Length of {text_type} Texts",
        xaxis_title="",
        yaxis_title="Length",
        showlegend=False,
    )
    fig.show()


df = read_from_csv()
show_text_length_boxplot(df, "origin", tokenize=True)
