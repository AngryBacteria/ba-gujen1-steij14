# TODO: properties graphs and tables of ggponc
import pandas as pd

from research.datacorpus.utils.utils_mongodb import get_collection


def save_to_csv():
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


def read_from_csv():
    df = pd.read_csv("ggponc_properties.csv", sep="|")
    return df


def type_barplot(df):
    import plotly.express as px

    type_counts = df["type"].value_counts()
    type_counts_df = pd.DataFrame(
        {"type": type_counts.index, "count": type_counts.values}
    )
    fig = px.pie(
        type_counts_df, values="count", names="type", title="Distribution of Types"
    )
    fig.show()


df = read_from_csv()
type_barplot(df)
