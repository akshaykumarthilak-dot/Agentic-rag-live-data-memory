"""
live_data.py
------------
Free live web-search functionality for Agentic RAG.

This module searches DuckDuckGo's public HTML search page
and returns structured search results.

No paid API key is required.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse, parse_qs


def search_live_data(query: str, max_results: int = 5) -> list:
    """
    Search the web using DuckDuckGo's public HTML search page.

    Returns a list of dictionaries containing:
    - title
    - url
    - snippet
    - source
    """

    if not query or not query.strip():
        return []

    query = query.strip()

    search_url = (
        "https://html.duckduckgo.com/html/?q="
        + quote(query)
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/153.0.0.0 Safari/537.36"
        )
    }

    try:

        response = requests.get(
            search_url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        results = []

        for result in soup.select(".result"):

            title_element = result.select_one(
                ".result__title"
            )

            link_element = result.select_one(
                ".result__a"
            )

            snippet_element = result.select_one(
                ".result__snippet"
            )

            if not title_element or not link_element:
                continue

            title = title_element.get_text(
                " ",
                strip=True
            )

            url = link_element.get(
                "href",
                ""
            )
            # DuckDuckGo often returns a redirect URL.
            # Extract the real destination URL.
            if url.startswith("//duckduckgo.com/l/"):

                parsed_url = urlparse(
                    "https:" + url
                )

                query_params = parse_qs(
                    parsed_url.query
                )

                real_url = query_params.get(
                    "uddg",
                    [url]
                )[0]

                url = real_url



            snippet = ""

            if snippet_element:
                snippet = snippet_element.get_text(
                    " ",
                    strip=True
                )

            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "live_web"
                }
            )

            if len(results) >= max_results:
                break

        return results

    except requests.RequestException as error:

        print(
            f"Live search request failed: {error}"
        )

        return []

    except Exception as error:

        print(
            f"Live search parsing failed: {error}"
        )

        return []