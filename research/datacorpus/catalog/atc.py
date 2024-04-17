import pandas as pd

from research.datacorpus.utils.utils_mongodb import upload_data_to_mongodb

# DATA SOURCE: Unknwon (Medication_Pharmacode_ATC.xlsx)

EXCEL_PATH = "F:\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\atc\\Medication_Pharmacode_ATC.xlsx"


def create_atc_db() -> None:
    """Parse the Excel file and create a MongoDB collection with the ATC codes."""
    column_names = [
        "pharmacode",
        "productcode",
        "text",
        "name",
        "dose",
        "form",
        "package_type",
        "package_size",
        "atc",
        "ingredient",
        "manufacturer",
        "btm",
        "primary_indication",
    ]
    data = pd.read_excel(EXCEL_PATH, header=0, dtype=str)
    data.columns = column_names
    # Remove empty rows and strip strings
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
    data = data[data["pharmacode"] != ""]
    data = data[data["name"] != ""]
    data = data[data["text"] != ""]
    data = data[data["atc"] != ""]

    # Fill NaN values with empty string
    data = data.fillna("")

    # Group by name and atc and keep other important fields as array
    grouped_df = (
        data.groupby(["name", "atc"])
        .agg(
            {
                "pharmacode": lambda x: x.tolist(),
                "productcode": lambda x: x.tolist(),
                "text": lambda x: x.tolist(),
                "form": lambda x: x.tolist(),
                "dose": lambda x: x.tolist(),
                "package_type": lambda x: x.tolist(),
                "ingredient": lambda x: x.tolist(),
                "manufacturer": lambda x: x.tolist(),
                "primary_indication": lambda x: x.tolist(),
            }
        )
        .reset_index()
    )
    grouped_df = grouped_df.to_dict(orient="records")

    # Upload to MongoDB
    upload_data_to_mongodb(grouped_df, "catalog", "atc", True, [])


# TODO: normalize the package type and doses
create_atc_db()
