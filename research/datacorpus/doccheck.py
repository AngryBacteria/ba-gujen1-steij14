import re

import requests
from bs4 import BeautifulSoup

from research.logger import logger


def get_all_fachgebiete():
    """Get links for all fachgebiete from doccheck.com"""

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
    """Get all tag links from doccheck.com"""
    response = requests.get(
        "https://flexikon.doccheck.com/de/Spezial:Tags?page=1&limit=20000"
    )
    if not response.ok:
        logger.error("Request failed")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    outer_container = soup.find("div", {"class": "dc-container"})
    container = outer_container.find("div", {"class": "is-flex column"})
    links = container.find_all("a")
    output = []
    for link in links:
        href = link.get("href")
        if href is None:
            continue
        output.append("https://flexikon.doccheck.com" + href)

    output = set(output)
    logger.debug(f"Found {len(output)} tags from doccheck.com")
    return output


def fetch_links_from_category_or_tag(url):
    """Fetch all links from a doccheck.com category page and return a list of links."""

    links = []
    current_url = url
    while current_url:
        response = requests.get(current_url)
        if not response.ok:
            logger.error(f"Request failed for url: {current_url}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        container = soup.find("div", {"class": "mw-category-generated"})

        # Extract all links
        links.extend([a["href"] for a in container.find_all("a", href=True)])
        # Find 'nächste Seite' link and update url, or set it to None to exit loop
        next_page_link = soup.find(
            "a", string=lambda text: text and "nächste Seite" in text
        )

        # filter out links that are not articles
        links = [
            link for link in links if not re.search(r"index.php", link, re.IGNORECASE)
        ]

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

    logger.debug(f"Found {len(links)} links from category {url}")
    return links


def get_all_links_from_doccheck(save_to_file=False):
    """Get all links from doccheck.com"""
    links = []
    fachgebiete = get_all_fachgebiete()
    for fachgebiet in fachgebiete:
        links.extend(fetch_links_from_category_or_tag(fachgebiet))

    links = list(set(links))

    if save_to_file:
        with open("doccheck_fachgebiete_artikel.txt", "w") as file:
            for entry in links:
                file.write(entry + "\n")

    return links
