from bs4 import Tag, ResultSet

ignore_list = [
    "Weblinks",
    "Literatur",
    "Einzelnachweise",
    "Siehe auch",
    "Referenzen",
    "Externe Links",
    "Quellen",
    "Weitere Informationen",
    "Zusätzliche Informationen",
    "Ähnliche Artikel",
    "Anmerkungen",
    "Audio",
]


def get_all_text(
        sections: ResultSet, full_text: bool
) -> str:
    """Get text for a wikipedia article from the content div.
    If full_text is set to True, the full text of the article is returned.
    Otherwise, only the introduction text is returned."""
    text = ""
    if not full_text:
        sections = [sections[0]]

    stop_processing = False
    for section in sections:
        if stop_processing:
            break

        for child in section.children:
            if (
                    child.name == "h2"
                    or child.name == "h3"
                    or child.name == "h4"
                    or child.name == "h5"
                    or child.name == "h6"
            ):
                if child.text.strip() in ignore_list:
                    stop_processing = True
                    break
                else:
                    text = text + "\n" + child.text + "\n"
            if child.name == "p":
                text += child.text + "\n"
            if child.name == "ul":
                text += process_ul(child)
            if child.name == "ol":
                text += process_ol(child)
            if child.name == "dl":
                text += process_dl(child)

    if text.startswith("- Wikidata:"):
        return ""
    return text


def process_ul(ul_element: Tag) -> str:
    """Recursively process a <ul> element and its children <li> elements to extract text."""
    list_text = ""
    for li in ul_element.find_all("li", recursive=False):
        list_text += "- " + li.text.strip() + "\n"
        for nested_ul in li.find_all("ul", recursive=False):
            list_text += process_ul(nested_ul)
    return list_text


def process_ol(ol_element: Tag) -> str:
    """Recursively process a <ol> element and its children <li> elements to extract text."""
    list_text = ""
    for li in ol_element.find_all("li", recursive=False):
        list_text += "- " + li.text.strip() + "\n"
        for nested_ol in li.find_all("ol", recursive=False):
            list_text += process_ol(nested_ol)
    return list_text


def process_dl(dl_element: Tag) -> str:
    """Recursively process a <dl> element and its children <dt> and <dd> elements to extract text."""
    list_text = ""
    for dt in dl_element.find_all("dt", recursive=False):
        list_text += "- " + dt.text.strip() + "\n"
        for dd in dt.find_all("dd", recursive=False):
            list_text += "  " + dd.text.strip() + "\n"
    return list_text