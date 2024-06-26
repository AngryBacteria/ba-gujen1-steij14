import os
import re

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient

from datacorpus.utils.scraping import (
    process_tags_to_text,
    remove_unwanted,
)
from shared.logger import logger

validation_articles = [
    "https://flexikon.doccheck.com/de/Rachen",
    "https://flexikon.doccheck.com/de/%C3%9Cberdruckbeatmung",
]


# link gathering
def get_all_fachgebiete_links():
    """Get the links for all upper level fachgebiete from doccheck.com"""

    response = requests.get("https://flexikon.doccheck.com/de/Kategorie:%C3%9Cbersicht")
    if not response.ok:
        logger.error("Request failed")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    container = soup.find("div", {"id": "mw-subcategories"})
    if container is None:
        logger.error("No mw-subcategories container found")
        return []

    links = container.find_all("a")
    output = []
    for link in links:
        href = link.get("href")
        if href is None:
            continue
        output.append("https://flexikon.doccheck.com" + href)

    output = set(output)
    logger.debug(f"Found {len(output)} fachgebiete from doccheck.com")
    return output


def get_all_tag_links():
    """Get all upper level tag links from doccheck.com"""
    response = requests.get(
        "https://flexikon.doccheck.com/de/Spezial:Tags?page=1&limit=20000"
    )
    if not response.ok:
        logger.error("Request failed")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    outer_container = soup.find("div", {"class": "dc-container"})
    if outer_container is None:
        logger.error("No dc-container found")
        return []

    container = outer_container.find("div", {"class": "is-flex column"})
    if container is None:
        logger.error("No is-flex column container found")
        return []

    links = container.find_all("a")
    output = []
    for link in links:
        href = link.get("href")
        if href is None:
            continue
        output.append(href)

    output = set(output)
    logger.debug(f"Found {len(output)} tags from doccheck.com")
    return output


def fetch_links_from_category_or_tag(url):
    """
    Fetch all links from a doccheck.com category or tag page and return a list of links.
    :param url: The url of the category or tag page.
    """

    links = []
    current_url = url
    while current_url:
        response = requests.get(current_url)
        if not response.ok:
            logger.error(f"Request failed for url: {current_url}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        container = soup.find("div", {"class": "mw-category-generated"})
        if container is None:
            logger.error("No mw-category-generated container found")
            break

        # Extract all links
        links.extend([a["href"] for a in container.find_all("a", href=True)])
        # Find 'nächste Seite' link and update url, or set it to None to exit loop
        next_page_link = soup.find(
            "a", string=lambda text: text and "nächste Seite" in text
        )

        if next_page_link:
            current_url = "https://flexikon.doccheck.com" + next_page_link["href"]
        else:
            current_url = None

    # filter out links that are not articles
    links = [link for link in links if not re.search(r"index.php", link, re.IGNORECASE)]
    # add domain to relative links
    links = ["https://flexikon.doccheck.com" + link for link in links]
    # filter out duplicates
    links = list(set(links))

    logger.debug(f"Found {len(links)} links from {url}")
    return links


def get_all_links_from_doccheck(save_to_file=True, read_from_file=False):
    """
    Get all relevant links from doccheck.com
    :param save_to_file: If True, save the links to a file.
    :param read_from_file: If True, read the links from a file.
    """
    if read_from_file:
        with open("doccheck_artikel.txt", "r") as file:
            return [line.strip() for line in file.readlines()]

    links = []
    fachgebiete = get_all_fachgebiete_links()
    for fachgebiet in fachgebiete:
        try:
            links.extend(fetch_links_from_category_or_tag(fachgebiet))
        except Exception as e:
            logger.error(f"Failed to fetch links for fachgebiet {fachgebiet}: {e}")

    tags = get_all_tag_links()
    for tag in tags:
        try:
            links.extend(fetch_links_from_category_or_tag(tag))
        except Exception as e:
            logger.error(f"Failed to fetch links for tag {tag}: {e}")

    links = list(set(links))

    if save_to_file:
        with open("doccheck_artikel.txt", "w") as file:
            for entry in links:
                file.write(entry + "\n")

    return links


# article scraping
def clean_doccheck_string(text: str) -> str:
    """
    Clean up text from doccheck.com by removing unwanted characters and whitespace.
    :param text: The text to clean.
    :return: The cleaned text.
    """
    text = text.replace("...", "")
    pattern = r"\[.*?\]"
    text = re.sub(pattern, "", text)
    text = text.strip()
    return text


def get_disciplines_and_tags_from_article(tags_category_container):
    """Get disciplines and tags from a doccheck article. Returns a tuple of lists."""
    if tags_category_container is None:
        return [], []

    try:
        disciplines = []
        tags = []
        if tags_category_container is not None:
            disciplines_element = tags_category_container.find(
                "div", {"class": "disciplines"}
            )
            if disciplines_element is not None and disciplines_element.text is not None:
                disciplines_text = disciplines_element.text.replace(
                    "Fachgebiete:", ""
                ).strip()
                disciplines = [
                    discipline.strip() for discipline in disciplines_text.split(",")
                ]
            tags_element = tags_category_container.find("div", {"class": "tags"})
            if tags_element is not None and tags_element.text is not None:
                tags_text = tags_element.text.replace("Stichworte:", "").strip()
                tags = [tag.strip() for tag in tags_text.split(",")]

        return disciplines, tags
    except Exception as e:
        logger.error(f"Failed to extract disciplines and tags: {e}")


def get_synonyms_and_english_title_from_article(first_p):
    """Get synonyms and english title from a doccheck article. Returns a tuple of lists."""
    if first_p is None:
        return [], []

    try:
        tags = []
        if first_p is not None:
            tags = first_p.find_all("i")
        synonyms = []
        english_title = []
        for tag in tags:
            if tag is not None and tag.text is not None:
                if tag.get_text(strip=True).startswith("Synonyme:"):
                    synonym_text = tag.get_text().replace("Synonyme:", "").strip()
                    synonyms = [s.strip() for s in synonym_text.split(",")]
                elif tag.get_text(strip=True).startswith("Synonym:"):
                    synonym_text = tag.get_text().replace("Synonym:", "").strip()
                    synonyms = [s.strip() for s in synonym_text.split(",")]
                elif tag.get_text(strip=True).startswith("Englisch:"):
                    english_title_text = tag.get_text().replace("Englisch:", "").strip()
                    english_title = [s.strip() for s in english_title_text.split(",")]

        return synonyms, english_title
    except Exception as e:
        logger.error(f"Failed to extract synonyms and english title: {e}")
        return [], []


def get_doccheck_article(url: str):
    """Get an article from doccheck.com by url. Returns a dictionary."""
    response = requests.get(url)
    if not response.ok:
        logger.error(f"Request failed for url: {url}")
        return None

    # get article container
    soup = BeautifulSoup(response.text, "html.parser")
    soup = remove_unwanted(soup)
    container = soup.find("div", {"id": "mw-content-text"})
    if container is None:
        logger.error(f"No mw-content-text container found for article: {url}")
        return None

    # get title
    title_element = soup.find("h1")
    if title_element is not None and title_element.text is not None:
        title = title_element.text.strip()
    else:
        logger.error(f"No title found for article: {url}")
        return None

    # get synonyms and english translation
    first_p = soup.find("p")
    synonyms, english_title = get_synonyms_and_english_title_from_article(first_p)
    # get disciplines and tags
    tags_category_container = soup.find("div", {"id": "categories"})
    disciplines, article_tags = get_disciplines_and_tags_from_article(
        tags_category_container
    )

    # get text from article elements
    articles = container.find_all("div", {"class": "collapsible-article"})
    text = ""
    for article in articles:
        header = article.find("div", {"class": "collapsible-heading"})
        content = article.find("div", {"class": "collapsible-content"})
        tags_header = header.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "dl"], recursive=False
        )
        tags_content = content.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "dl"], recursive=False
        )
        tags = tags_header + tags_content
        # Filter out tags with that contain class mw-empty-elt
        tags = [tag for tag in tags if "mw-empty-elt" not in (tag.get("class") or [])]
        text += process_tags_to_text(tags)

    # return cleaned text
    if text is None or text == "":
        logger.warning(f"No text found for article: {url}")
        return None
    text = clean_doccheck_string(text)
    return {
        "title": title,
        "english_title": english_title,
        "synonyms": synonyms,
        "text": text,
        "link": url,
        "tags": article_tags,
        "disciplines": disciplines,
    }


def build_doccheck_corpus(from_file=False):
    """
    Build a corpus of articles from doccheck.com
    :param from_file: If True, read links from file.
    """
    load_dotenv()
    client = MongoClient(os.getenv("MONGO_URL"))
    db = client.get_database("web")
    doccheck_collection = db.get_collection("doccheck")
    doccheck_collection.create_index("link", unique=True)

    links = get_all_links_from_doccheck(read_from_file=from_file)
    for link in links:
        try:
            existing_doc = doccheck_collection.find_one({"link": link})
            if existing_doc is None:
                article = get_doccheck_article(link)
                if article is not None:
                    doccheck_collection.insert_one(article)
                    logger.debug(f"Inserted Doccheck article: {link}")
            else:
                logger.debug(f"Doccheck article already exists: {link}")
                continue
        except Exception as e:
            logger.error(f"Failed to fetch article from link {link}: {e}")

    logger.info("Finished building doccheck corpus")
    client.close()


if __name__ == "__main__":
    build_doccheck_corpus(from_file=True)
