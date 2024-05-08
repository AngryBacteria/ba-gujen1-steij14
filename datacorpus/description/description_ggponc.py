import pandas as pd
from pandas import DataFrame

from datacorpus.utils.mongodb import get_collection
from shared.model_utils import patch_tokenizer_with_template


def save_to_csv() -> None:
    """
    Save the ggponc mongodb to a trimmed down csv file
    :return: None
    """
    ggonc_collection = get_collection("corpus", "ggponc_short")
    ggonc_cursor = ggonc_collection.find({})

    formatted_annotations = []
    for doc in ggonc_cursor:
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
    df.to_csv("ggponc_description.csv", index=False, sep="|")


def read_from_csv() -> DataFrame:
    """
    Read the trimmed down ggponc properties csv file
    :return: ggponc properties dataframe
    """
    df = pd.read_csv("ggponc_description.csv", sep="|", na_filter=False)
    return df


def show_extraction_types_pieplot(df: DataFrame) -> None:
    """
    Plot the distribution of the annotation types
    :param df: The dataframe to plot from
    :return: None (a plot is displayed)
    """
    import plotly.express as px

    type_counts = df["type"].value_counts()
    type_counts_df = pd.DataFrame(
        {"type": type_counts.index, "count": type_counts.values}
    )
    fig = px.pie(
        type_counts_df,
        values="count",
        names="type",
        title="Distribution of annotation types",
    )
    fig.show()


def show_top_labels_barplot(df: DataFrame, annotation_types=None) -> None:
    """
    Plot the distribution of the top 20 extraction labels
    :param df: The dataframe to plot from
    :param annotation_types: The extraction types to use for the plot
    :return: None (a plot is displayed)
    """
    if annotation_types is None:
        annotation_types = ["TREATMENT", "DIAGNOSIS", "MEDICATION"]
    import plotly.express as px

    filtered_df = df[df["type"].isin(annotation_types)]
    type_counts = filtered_df["text"].value_counts()
    top_20_counts = type_counts.head(20)
    top_20_counts_df = pd.DataFrame(
        {"text": top_20_counts.index, "count": top_20_counts.values}
    )

    desired_types_string = ", ".join(annotation_types)
    total_number = len(filtered_df)

    fig = px.bar(
        top_20_counts_df,
        x="text",
        y="count",
        title=f"Distribution of Top 20 annotation labels (n = {total_number}, types = {desired_types_string})",
    )
    fig.show()


def get_paragraph_lengths(df, tokenize=False) -> tuple:
    """
    Get the max, min, average and median length of the paragraphs
    :param df: The dataframe to calculate from. Needs to have an "origin" column
    :param tokenize: If the lengths should be calculated by tokens or not
    :return: The max, min, average and median length of the paragraphs
    """
    lengths = []

    if tokenize:

        tokenizer = patch_tokenizer_with_template()
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


def show_paragraph_lengths_boxplot(df: DataFrame, tokenize=False) -> None:
    import plotly.express as px

    # calculate the lengths
    lengths = []
    if tokenize:

        tokenizer = patch_tokenizer_with_template()
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
    fig = px.box(y=lengths)
    fig.update_layout(
        title="Paragraph Lengths",
        xaxis_title="",
        yaxis_title="Length",
        showlegend=False,
    )
    fig.show()
