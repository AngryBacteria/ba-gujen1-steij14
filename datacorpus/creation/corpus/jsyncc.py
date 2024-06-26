import xml.etree.ElementTree as ET

from shared.mongodb import upload_data_to_mongodb

corpus_path = "jsyncc\\corpus.xml"

# PAPER: https://aclanthology.org/L18-1201/
# DATA SOURCE: https://github.com/JULIELab/jsyncc


def find_with_default(element, tag, default_value="") -> str:
    """Attempt to find the tag in the element, return text or a default value if not found or empty."""
    found_element = element.find(tag)
    if (
        found_element is not None
        and found_element.text is not None
        and found_element.text.strip()
    ):
        return found_element.text.strip()
    return default_value


def get_jsyncc_data() -> list[dict[str, str]]:
    """
    Get the jsyncc data from the XML file.
    :return: Data as list of dictionaries.
    """

    tree = ET.parse(corpus_path)
    root = tree.getroot()
    documents = root.findall(".//document")
    # Convert XML to dictionary
    corpus = []
    for document in documents:
        doc_id = find_with_default(document, "id")
        id_long = find_with_default(document, "idLong")
        text = find_with_default(document, "text")
        doc_type = find_with_default(document, "type")
        heading = find_with_default(document, "heading")
        source = find_with_default(document, "source")

        corpus.append(
            {
                "doc_id": doc_id,
                "id_long": id_long,
                "text": text,
                "doc_type": doc_type,
                "heading": heading,
                "source": source,
            }
        )

    return corpus


def build_jsyncc_db() -> None:
    """
    Build the jsyncc database.
    :return: None
    """
    corpus = get_jsyncc_data()
    upload_data_to_mongodb(corpus, "corpus", "jsyncc", True, [])


if __name__ == "__main__":
    build_jsyncc_db()
