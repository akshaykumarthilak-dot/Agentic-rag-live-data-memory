"""
tools.py
--------
Live-data tools for Agentic RAG.
"""

import requests

from live_data import search_live_data


def wikipedia_search(query: str, sentences: int = 3) -> str:
    """
    Fetches a short summary of a topic from Wikipedia.
    """

    try:

        resp = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + query.replace(" ", "_"),
            timeout=8,
        )

        if resp.status_code != 200:
            return f"No live Wikipedia result found for '{query}'."

        data = resp.json()

        extract = data.get("extract", "")

        if not extract:
            return f"No summary available for '{query}'."

        parts = extract.split(". ")

        return (
            ". ".join(parts[:sentences]).strip()
            + ("." if not extract.endswith(".") else "")
        )

    except requests.RequestException as e:

        return f"Live search failed: {e}"


def get_current_weather(city: str) -> str:
    """
    Fetches live current weather using Open-Meteo.
    """

    try:

        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1
            },
            timeout=8,
        ).json()

        if not geo.get("results"):
            return f"Could not find location '{city}'."

        loc = geo["results"][0]

        lat = loc["latitude"]
        lon = loc["longitude"]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True
            },
            timeout=8,
        ).json()

        cw = weather.get(
            "current_weather",
            {}
        )

        if not cw:
            return f"Weather data unavailable for {city}."

        return (
            f"Current weather in "
            f"{loc.get('name', city)}: "
            f"{cw.get('temperature')}°C, "
            f"wind {cw.get('windspeed')} km/h, "
            f"recorded at {cw.get('time')}."
        )

    except requests.RequestException as e:

        return f"Live weather fetch failed: {e}"


def live_web_search(query: str) -> str:
    """
    Search the live web using DuckDuckGo.

    Returns a compact text representation
    of the search results.
    """

    results = search_live_data(
        query,
        max_results=5
    )

    if not results:
        return f"No live web results found for '{query}'."

    formatted_results = []

    for i, result in enumerate(
        results,
        start=1
    ):

        formatted_results.append(
            f"Result {i}:\n"
            f"Title: {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Snippet: {result['snippet']}"
        )

    return "\n\n".join(
        formatted_results
    )


# ---------------------------------------------------------
# TOOL REGISTRY
# ---------------------------------------------------------

TOOL_REGISTRY = {

    "wikipedia_search": {
        "fn": wikipedia_search,
        "description": (
            "Look up a current factual summary of a "
            "topic/entity from Wikipedia. "
            "Input: a search term."
        ),
    },

    "get_current_weather": {
        "fn": get_current_weather,
        "description": (
            "Get real-time current weather for a "
            "named city. Input: a city name."
        ),
    },

    "live_web_search": {
        "fn": live_web_search,
        "description": (
            "Search the live web for current information. "
            "Use this when the answer may have changed recently "
            "or is not available in the local knowledge base. "
            "Input: a search query."
        ),
    },
}