# TODO: properties graphs and tables of bronco
import pandas as pd
from pandas import DataFrame

from research.datacorpus.creation.utils.utils_mongodb import get_collection


def save_to_csv() -> None:
    """
    Save the ggponc mongodb to trimmed down csv file
    :return: None
    """
    bronco_collection = get_collection("corpus", "bronco")
    bronco_cursor = bronco_collection.find({})

    formatted_annotations = []
    for doc in bronco_cursor:
        if doc["type"] == "NA":
            formatted_annotations.append(
                {
                    "type": doc["type"],
                    "origin": doc["origin"],
                    "text": "NA",
                    "normalization": "NA",
                    "localisation": "NA",
                    "level_of_truth": "NA",
                }
            )
        else:
            for index, text in enumerate(doc["text"]):

                normalization = doc["normalizations"][index][0]["normalization"].split(
                    ":"
                )[1]

                level_of_truth = "true"
                localisation = "NA"
                if len(doc["attributes"]) > index:
                    for attribute in doc["attributes"][index]:
                        if attribute["attribute_label"] == "LevelOfTruth":
                            level_of_truth = attribute["attribute"]
                        if attribute["attribute_label"] == "Localisation":
                            localisation = attribute["attribute"]

                formatted_annotations.append(
                    {
                        "type": doc["type"],
                        "origin": doc["origin"],
                        "text": text,
                        "normalization": normalization,
                        "level_of_truth": level_of_truth,
                        "localisation": localisation,
                    }
                )

    df = pd.DataFrame(formatted_annotations)
    df.to_csv("bronco_description.csv", index=False, sep="|")


def read_from_csv() -> DataFrame:
    """
    Read the ggponc properties csv file
    :return: ggponc properties dataframe
    """
    df = pd.read_csv("bronco_description.csv", sep="|", na_filter=False)
    return df


def show_extraction_types_pieplot(df: DataFrame) -> None:
    """
    Plot the distribution of the extraction types
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
        title="Distribution of extraction types",
    )
    fig.show()


def show_truth_levels_pieplot(df: DataFrame, desired_types=None) -> None:
    """
    Plot the level_of_truth types distribution
    :param df: The dataframe to plot from
    :param desired_types: The extraction types to use for the plot (mostly you don't want na)
    :return: None (a plot is displayed)
    """
    import plotly.express as px

    if desired_types is None:
        desired_types = ["TREATMENT", "DIAGNOSIS", "MEDICATION"]
    filtered_df = df[df["type"].isin(desired_types)]

    type_counts = filtered_df["level_of_truth"].value_counts()
    type_counts_df = pd.DataFrame(
        {"level_of_truth": type_counts.index, "count": type_counts.values}
    )
    fig = px.pie(
        type_counts_df,
        values="count",
        names="level_of_truth",
        title="Distribution of levels of truth",
    )
    fig.show()


def show_localisation_pieplot(df: DataFrame, desired_types=None) -> None:
    """
    Plot the localisation types distribution
    :param df: The dataframe to plot from
    :param desired_types: The extraction types to use for the plot (mostly you don't want na)
    :return: None (a plot is displayed)
    """
    import plotly.express as px

    if desired_types is None:
        desired_types = ["TREATMENT", "DIAGNOSIS", "MEDICATION"]
    filtered_df = df[df["type"].isin(desired_types)]

    type_counts = filtered_df["localisation"].value_counts()
    type_counts_df = pd.DataFrame(
        {"level_of_truth": type_counts.index, "count": type_counts.values}
    )
    fig = px.pie(
        type_counts_df,
        values="count",
        names="level_of_truth",
        title="Distribution of Localisation labels",
    )
    fig.show()


def show_text_lengths_boxplot(df: DataFrame, tokenize=False) -> None:
    """
    Plot the length of the origin texts as a boxplot. Either count tokens or characters
    :param df: The dataframe to plot from
    :param tokenize: If the lengths should be calculated by tokens or not
    :return: None (a plot is displayed)
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
    fig = px.box(y=lengths)
    fig.update_layout(
        title="Length of Origin Texts",
        xaxis_title="",
        yaxis_title="Length",
        showlegend=False,
    )
    fig.show()


def get_text_lengths(df, tokenize=False) -> tuple:
    """
    Get the max, min, average and median length of the origin texts
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


def show_top_labels_barplot(df: DataFrame, desired_types=None) -> None:
    """
    Plot the distribution of the top 20 extraction labels
    :param df: The dataframe to plot from
    :param desired_types: The extraction types to use for the plot
    :return: None (a plot is displayed)
    """
    if desired_types is None:
        desired_types = ["TREATMENT", "DIAGNOSIS", "MEDICATION"]
    import plotly.express as px

    filtered_df = df[df["type"].isin(desired_types)]
    type_counts = filtered_df["text"].value_counts()
    top_20_counts = type_counts.head(20)
    top_20_counts_df = pd.DataFrame(
        {"text": top_20_counts.index, "count": top_20_counts.values}
    )

    desired_types_string = ", ".join(desired_types)
    total_number = len(filtered_df)

    fig = px.bar(
        top_20_counts_df,
        x="text",
        y="count",
        title=f"Distribution of Top 20 Extraction labels (n = {total_number}, types = {desired_types_string})",
    )
    fig.show()


def show_top_normalizations_barplot(df: DataFrame, desired_types=None) -> None:
    """
    Plot the distribution of the top 20 normalization labels
    :param df: The dataframe to plot from
    :param desired_types: The extraction types to use for the plot
    :return: None (a plot is displayed)
    """
    if desired_types is None:
        desired_types = ["TREATMENT", "DIAGNOSIS", "MEDICATION"]
    import plotly.express as px

    filtered_df = df[df["type"].isin(desired_types)]
    type_counts = filtered_df["normalization"].value_counts()
    top_20_counts = type_counts.head(20)
    top_20_counts_df = pd.DataFrame(
        {"normalization": top_20_counts.index, "count": top_20_counts.values}
    )

    desired_types_string = ", ".join(desired_types)
    total_number = len(filtered_df)

    fig = px.bar(
        top_20_counts_df,
        x="normalization",
        y="count",
        title=f"Distribution of the Top 20 Normalizations (n = {total_number}, types = {desired_types_string})",
    )
    fig.show()


save_to_csv()
df_main = read_from_csv()
show_truth_levels_pieplot(df_main)
