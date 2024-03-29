from bs4 import Tag

ignore_list = [
    "Weblinks",
    "Literatur",
    "Fachliteratur",
    "Leitlinien",
    "Einzelnachweise",
    "Siehe auch",
    "Referenzen",
    "Externe Links",
    "Quellen",
    "Filme",
    "Archivalien",
    "Weitere Informationen",
    "Zusätzliche Informationen",
    "Ähnliche Artikel",
    "Anmerkungen",
    "Audio",
    "Bildquelle",
    "Auskultationsgeräusch",
]


def process_tags_to_text(tags: list[Tag], full_text=True) -> str:
    if not full_text:
        tags = [tags[0]]

    text = ""
    for child in tags:
        if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            if child.text.strip() in ignore_list:
                break
            else:
                text = text + "\n" + child.text.strip() + "\n"
        elif child.name == "p":
            text += child.text.strip() + "\n"
        elif child.name == "ul":
            text += process_ul(child)
        elif child.name == "ol":
            text += process_ol(child)
        elif child.name == "dl":
            text += process_dl(child)
    return text


def process_ul(ul_element: Tag) -> str:
    """Process a <ul> element and its children <li> elements to extract text."""
    list_text = ""
    for li in ul_element.find_all("li", recursive=False):
        list_text += "- " + li.text.strip() + "\n"
    return list_text


def process_ol(ol_element: Tag) -> str:
    """Process a <ol> element and its children <li> elements to extract text."""
    list_text = ""
    for li in ol_element.find_all("li", recursive=False):
        list_text += "- " + li.text.strip() + "\n"
    return list_text


def process_dl(dl_element: Tag) -> str:
    """Recursively process a <dl> element and its children <dt> and <dd> elements to extract text."""
    list_text = ""
    for element in dl_element.find_all(["dt", "dd"]):
        if element.name == "dt":
            list_text += "- " + element.text.strip() + "\n"
        if element.name == "dd":
            list_text += "  " + element.text.strip() + "\n"

    return list_text
