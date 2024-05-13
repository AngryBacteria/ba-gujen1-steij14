import pandas as pd
from pandas import DataFrame

from shared.mongodb import get_collection
from shared.model_utils import patch_tokenizer_with_template


def save_to_csv() -> None:
    """
    Save the ggponc mongodb to a trimmed down csv file
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
    Read the ggponc properties from the trimmed down csv file
    :return: ggponc properties dataframe
    """
    df = pd.read_csv("bronco_description.csv", sep="|", na_filter=False)
    return df


def show_annotation_types_pieplot(df: DataFrame) -> None:
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


def show_truth_levels_pieplot(df: DataFrame, annotation_types=None) -> None:
    """
    Plot the level_of_truth types distribution
    :param df: The dataframe to plot from
    :param annotation_types: The annotation types to use for the plot (mostly you don't want na)
    :return: None (a plot is displayed)
    """
    import plotly.express as px

    if annotation_types is None:
        annotation_types = ["TREATMENT", "DIAGNOSIS", "MEDICATION"]
    filtered_df = df[df["type"].isin(annotation_types)]

    type_counts = filtered_df["level_of_truth"].value_counts()
    type_counts_df = pd.DataFrame(
        {"level_of_truth": type_counts.index, "count": type_counts.values}
    )
    fig = px.pie(
        type_counts_df,
        values="count",
        names="level_of_truth",
        title="Distribution of levels of truth attributes",
    )
    fig.show()


def show_localisation_pieplot(df: DataFrame, annotation_types=None) -> None:
    """
    Plot the localisation types distribution
    :param df: The dataframe to plot from
    :param annotation_types: The annotation types to use for the plot (mostly you don't want na)
    :return: None (a plot is displayed)
    """
    import plotly.express as px

    if annotation_types is None:
        annotation_types = ["TREATMENT", "DIAGNOSIS", "MEDICATION"]
    filtered_df = df[df["type"].isin(annotation_types)]

    type_counts = filtered_df["localisation"].value_counts()
    type_counts_df = pd.DataFrame(
        {"level_of_truth": type_counts.index, "count": type_counts.values}
    )
    fig = px.pie(
        type_counts_df,
        values="count",
        names="level_of_truth",
        title="Distribution of the localisation attribute",
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
        tokenizer = patch_tokenizer_with_template()
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

        tokenizer = patch_tokenizer_with_template()
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


def show_top_labels_barplot(df: DataFrame, annotation_types=None) -> None:
    """
    Plot the distribution of the top 20 annotation labels for a specific annotation type
    :param df: The dataframe to plot from
    :param annotation_types: The annotation types to use for the plot
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


def show_top_normalizations_barplot(df: DataFrame, annotation_types=None) -> None:
    """
    Plot the distribution of the top 20 normalization labels
    :param df: The dataframe to plot from
    :param annotation_types: The annotation types to use for the plot
    :return: None (a plot is displayed)
    """
    if annotation_types is None:
        annotation_types = ["TREATMENT", "DIAGNOSIS", "MEDICATION"]
    import plotly.express as px

    filtered_df = df[df["type"].isin(annotation_types)]
    type_counts = filtered_df["normalization"].value_counts()
    top_20_counts = type_counts.head(20)
    top_20_counts_df = pd.DataFrame(
        {"normalization": top_20_counts.index, "count": top_20_counts.values}
    )

    desired_types_string = ", ".join(annotation_types)
    total_number = len(filtered_df)

    fig = px.bar(
        top_20_counts_df,
        x="normalization",
        y="count",
        title=f"Distribution of the Top 20 normalizations (n = {total_number}, types = {desired_types_string})",
    )
    fig.show()


def show_distribution_na_notna():
    """
    Show the distribution of NA and Not NA in the origin texts
    """
    bronco_collection = get_collection("corpus", "bronco")
    bronco_cursor = bronco_collection.find({})
    bronco_docs = list(bronco_cursor)

    bronnco_df = pd.DataFrame(bronco_docs)
    bronnco_df = (
        bronnco_df.groupby(["origin"]).agg({"type": lambda x: x.tolist()}).reset_index()
    )

    # make the items inside the type array unique
    bronnco_df["type"] = bronnco_df["type"].apply(lambda x: list(set(x)))
    value_counts = bronnco_df["type"].value_counts()

    # get distribution between NA and all others
    is_na = 0
    is_not_na = 0
    for index, value in value_counts.items():
        if index == ["NA"]:
            is_na += value
        else:
            is_not_na += value

    # pie plot
    import plotly.express as px

    fig = px.pie(
        names=["NA", "Not NA"],
        values=[is_na, is_not_na],
        title="Distribution of NA and Not NA in Origin Texts",
    )
    fig.show()
