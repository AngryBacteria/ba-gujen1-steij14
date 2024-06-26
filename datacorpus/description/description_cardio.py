import ast

import pandas as pd
from pandas import DataFrame

from shared.decoder_utils import load_tokenizer_with_template, count_tokens
from shared.mongodb import get_collection


# Collection of functions to analyze the cardio dataset.


def save_to_csv() -> None:
    """
    Save the cardio mongodb collection to a trimmed down csv file
    :return: None a
    """

    cardio_collection = get_collection("corpus", "cardio")
    cardio_cursor = cardio_collection.find({})

    print(cardio_collection.count_documents({}))

    formatted_annotations = []
    for doc in cardio_cursor:
        for anno in doc["annotations"]:
            if anno["type"] == "NA":
                formatted_annotations.append(
                    {
                        "document": doc["document"],
                        "type": anno["type"],
                        "origin": anno["origin"],
                        "text": "NA",
                        "attributes": [],
                    }
                )
            else:
                for text in anno["text"]:
                    formatted_annotations.append(
                        {
                            "document": doc["document"],
                            "type": anno["type"],
                            "origin": anno["origin"],
                            "text": text,
                            "attributes": anno["attributes"],
                        }
                    )

    df = pd.DataFrame(formatted_annotations)
    df.to_csv("cardio_description.csv", sep="|", index=False)


def save_full_text_to_csv() -> None:
    """
    Save the cardio mongodb collection (only full_text) to a trimmed down csv file
    :return: None a csv file is created
    """

    cardio_collection = get_collection("corpus", "cardio")
    cardio_cursor = cardio_collection.find({})

    formatted_annotations = []
    for doc in cardio_cursor:
        formatted_annotations.append(
            {"document": doc["document"], "origin": doc["full_text"]}
        )

    df = pd.DataFrame(formatted_annotations)
    df.to_csv("cardio_full_text.csv", sep="|", index=False)


def read_from_csv(file_name="cardio_description.csv") -> DataFrame:
    """
    Read the cardio properties csv file
    :return: cardio properties dataframe
    """
    df = pd.read_csv(file_name, sep="|", na_filter=False)
    return df


def show_type_pieplot(df: DataFrame) -> None:
    """
    Plot the distribution of the annotation types based on the number of characters in 'origin'.
    :param df: The dataframe to plot from
    :return: None (a plot is displayed)
    """
    import plotly.express as px

    characters_with = 0
    characters_without = 0
    for _, row in df.iterrows():
        if row["type"] == "NA":
            characters_without += len(row["origin"])
        else:
            characters_with += len(row["origin"])
    data = {
        "type": ["With annotation", "Without annotation"],
        "count": [characters_with, characters_without],
    }
    type_counts_df = pd.DataFrame(data)

    fig = px.pie(
        type_counts_df,
        values="count",
        names="type",
        title="Distribution of annotation types in Cardio Corpus based on character count",
    )
    fig.show()


def get_number_of_annotations(df: DataFrame) -> int:
    """
    Get the number of annotations of type 'MEDICATION'
    :param df: The dataframe to analyze
    :return: The number of annotations of type 'MEDICATION'
    """
    count = 0
    for _, row in df.iterrows():
        if row["type"] == "MEDICATION":
            count += 1
    return count


def get_number_of_attributes(df: DataFrame) -> dict[str, int]:
    """
    Get the number of attributes for each attribute type
    :param df: The dataframe to analyze
    :return: Number of attributes for each attribute type
    """
    count_frequency = 0
    count_strength = 0
    count_duration = 0
    count_form = 0

    for _, row in df.iterrows():
        attributes = ast.literal_eval(row["attributes"])
        for attribute in attributes:
            for value in attribute:
                if value["attribute_label"] == "FREQUENCY":
                    count_frequency += 1
                elif value["attribute_label"] == "STRENGTH":
                    count_strength += 1
                elif value["attribute_label"] == "DURATION":
                    count_duration += 1
                elif value["attribute_label"] == "FORM":
                    count_form += 1

    return {
        "count_frequency": count_frequency,
        "count_strength": count_strength,
        "count_duration": count_duration,
        "count_form": count_form,
    }


def paragraph_lengths(df, tokenize=False) -> tuple:
    """
    Get the max, min, average and median length of the paragraphs
    :param df: The dataframe to calculate from. Needs to have an "origin" column
    :param tokenize: If the lengths should be calculated by tokens or not
    :return: The max, min, average and median length of the paragraphs
    """
    lengths = []
    if tokenize:

        tokenizer = load_tokenizer_with_template()
        for index, row in df.iterrows():
            paragraph_length = count_tokens([row["origin"]], tokenizer)
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


def show_lengths_boxplot(df: DataFrame, tokenize=False) -> None:
    """
    Plot the distribution of paragraph lengths
    :param df: The dataframe to plot from
    :param tokenize: Whether to calculate lengths based on tokenization
    :return: None (a plot is displayed)
    """
    import plotly.express as px

    # calculate the lengths
    lengths = []
    if tokenize:
        tokenizer = load_tokenizer_with_template()
        for index, row in df.iterrows():
            paragraph_length = count_tokens([row["origin"]], tokenizer)
            lengths.append(paragraph_length)
    else:
        for index, row in df.iterrows():
            paragraph_length = len(row["origin"])
            lengths.append(paragraph_length)

    # display the boxplot
    fig = px.box(y=lengths, title="Paragraph Lengths in Cardio Corpus")
    fig.update_layout(xaxis_title="", yaxis_title="Length", showlegend=False)
    fig.show()


def show_types_per_document_plot(df: DataFrame) -> None:
    """
    Analyze and visualize the count of annotation types per unique document.
    :param df: The dataframe to plot from
    :return: None (a plot is displayed)
    """

    import plotly.express as px

    # group per type
    type_counts = df.groupby(["document", "type"]).size().reset_index(name="count")

    # barplot
    fig = px.bar(
        type_counts,
        x="document",
        y="count",
        color="type",
        title="Type Counts per Document in Cardio Corpus",
        labels={"count": "Count of Types", "document": "Document ID", "type": "Type"},
    )
    fig.update_layout(
        xaxis_title="Document ID", yaxis_title="Count of Types", showlegend=True
    )
    fig.show()

    print(type_counts)


def show_medication_counts_per_document_plot(df: DataFrame) -> None:
    """
    Analyze and visualize the count of 'Medication' type per unique document.
    :param df: The dataframe to analyze
    :return: None (a plot is displayed)
    """
    import numpy as np
    import plotly.express as px

    # only type medication
    medication_df = df[df["type"] == "MEDICATION"]

    # Gruppiere nach 'document' und zähle die Anzahl der Vorkommen von 'Medication'
    medication_counts = (
        medication_df.groupby("document").size().reset_index(name="count")
    )

    max_count = medication_counts["count"].max()
    min_count = medication_counts["count"].min()
    median_count = medication_counts["count"].median()
    average_count = medication_counts["count"].mean()

    # Visualisiere die Ergebnisse als Balkendiagramm
    fig = px.bar(
        medication_counts,
        x="document",
        y="count",
        title="Medication Counts per Document in Cardio Corpus",
        labels={"count": "Count of Medication", "document": "Document ID"},
    )
    fig.update_layout(
        xaxis_title="Document ID", yaxis_title="Count of Medication", showlegend=False
    )
    fig.show()

    print(f"Max Count: {max_count}")
    print(f"Min Count: {min_count}")
    print(f"Median Count: {median_count}")
    print(f"Average Count: {np.round(average_count, 2)}")


def show_top_labels_barplot(df: DataFrame, annotation_types=None) -> None:
    """
    Plot the distribution of the top 20 annotation labels
    :param df: The dataframe to plot from
    :param annotation_types: The extraction types to use for the plot
    :return: None (a plot is displayed)
    """
    if annotation_types is None:
        annotation_types = ["MEDICATION"]
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


if __name__ == "__main__":
    save_to_csv()
    save_full_text_to_csv()
    df = read_from_csv("cardio_description.csv")
    df2 = read_from_csv("cardio_full_text.csv")
    show_type_pieplot(df)
    show_lengths_boxplot(df2, tokenize=True)
    print(get_number_of_annotations(df))
    print(get_number_of_attributes(df))
