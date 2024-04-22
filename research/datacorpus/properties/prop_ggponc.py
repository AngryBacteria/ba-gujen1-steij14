# TODO: properties graphs and tables of ggponc
import pandas as pd
from pandas import DataFrame

from research.datacorpus.utils.utils_mongodb import get_collection


def save_to_csv() -> None:
    ggonc_collection = get_collection("corpus", "ggponc_short")
    ggonc_cursor = ggonc_collection.find({})

    formatted_annotations = []
    for doc in ggonc_cursor:
        for anno in doc["annotations"]:
            for text in anno["text"]:
                formatted_annotations.append(
                    {"type": anno["type"], "origin": anno["origin"], "text": text}
                )

    df = pd.DataFrame(formatted_annotations)
    df.to_csv("ggponc_properties.csv", index=False, sep="|")


def read_from_csv() -> DataFrame:
    df = pd.read_csv("ggponc_properties.csv", sep="|")
    return df


def type_pieplot(df: DataFrame):
    import plotly.express as px

    type_counts = df["type"].value_counts()
    type_counts_df = pd.DataFrame(
        {"type": type_counts.index, "count": type_counts.values}
    )
    fig = px.pie(
        type_counts_df, values="count", names="type", title="Distribution of Types"
    )
    fig.show()


def text_barplot(df: DataFrame, desired_types=None):
    if desired_types is None:
        desired_types = ["TREATMENT", "DIAGNOSIS", "None", "MEDICATION"]
    import plotly.express as px

    filtered_df = df[df["type"].isin(desired_types)]
    type_counts = filtered_df["text"].value_counts()
    top_20_counts = type_counts.head(20)  # Get the top 20 counts
    top_20_counts_df = pd.DataFrame(
        {"text": top_20_counts.index, "count": top_20_counts.values}
    )
    fig = px.bar(
        top_20_counts_df,
        x="text",
        y="count",
        title="Distribution of Top 20 Extractions",
    )
    fig.show()


def paragraph_lengths(df):
    # iterate over the rows of the dataframe
    lengths = []
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

    print(f"Max: {max_length}, Min: {min_length}, Average: {avg_length}, Median: {median}")
    return max_length, min_length, avg_length, median


df_main = read_from_csv()