import pandas as pd

from shared.mongodb import upload_data_to_mongodb

# DATA SOURCE: Unknown origin. (Medication_Pharmacode_ATC.xlsx)

# Path to the Excel file
EXCEL_PATH = "S:\\documents\\onedrive_bfh\\OneDrive - Berner Fachhochschule\\Dokumente\\UNI\\Bachelorarbeit\\datensets\\catalog\\atc\\Medication_Pharmacode_ATC.xlsx"


def create_atc_db() -> None:
    """Parse the Excel file and create a MongoDB collection with the ATC codes."""
    column_names = [
        "pharmacode",
        "productcode",
        "text",
        "title",
        "dose",
        "form",
        "package_type",
        "package_size",
        "code",
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
    data = data[data["title"] != ""]
    data = data[data["text"] != ""]
    data = data[data["code"] != ""]

    # Fill NaN values with empty string
    data = data.fillna("")

    # Group by title and atc_code and keep other important fields as array
    grouped_df = (
        data.groupby(["title", "code"])
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


create_atc_db()
