import pandas as pd
from pandas import DataFrame

from datacorpus.utils.mongodb import get_collection


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
                    {
                        "document": doc["document"],
                        "type": anno["type"],
                        "origin": anno["origin"],
                        "text": "NA",
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
                        }
                    )

    df = pd.DataFrame(formatted_annotations)
    df.to_csv("cardio_description.csv", sep="|", index=False)


def save_full_text_to_csv() -> None:
    """
    Save the cardio mongodb (only document and its full_text) to trimmed down csv file
    :return: None
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


def read_from_csv(str="cardio_description.csv") -> DataFrame:
    """
    Read the cardio properties csv file
    :return: cardio properties dataframe
    """
    df = pd.read_csv(str, sep="|", na_filter=False)
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
        type_counts_df,
        values="count",
        names="type",
        title="Distribution of Types in Cardio Corpus",
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
    fig.update_layout(xaxis_title="", yaxis_title="Length", showlegend=False)
    fig.show()


def analyze_types_per_document(df: DataFrame) -> None:
    """
    Analyze and visualize the count of types per unique document.
    :param df: The dataframe to plot from
    :return: None
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


def analyze_medication_counts_per_document(df: DataFrame) -> None:
    """
    Analyze and visualize the count of 'Medication' type per unique document.
    :param df: The dataframe to analyze
    :return: None
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


def show_top_labels_barplot(df: DataFrame, desired_types=None) -> None:
    """
    Plot the distribution of the top 20 extraction labels
    :param df: The dataframe to plot from
    :param desired_types: The extraction types to use for the plot
    :return: None (a plot is displayed)
    """
    if desired_types is None:
        desired_types = ["MEDICATION"]
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
        title=f"Distribution of Top 20 Extractions (n = {total_number}, types = {desired_types_string})",
    )
    fig.show()


save_to_csv()
save_full_text_to_csv()
df_main = read_from_csv()
df_text = read_from_csv("cardio_full_text.csv")
# type_pieplot(df_main)
# analyze_medication_counts_per_document(df_main)
# paragraph_lengths(df_text, tokenize=True)
plot_lengths_boxplot(df_text, tokenize=True)
plot_lengths_boxplot(df_main, tokenize=True)

# show_top_labels_barplot(df_main)
