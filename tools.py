"""
tools.py
--------
The "live data" half of Agentic RAG.

These are plain Python functions with no LLM involved — the agent decides
WHEN to call them, but the functions themselves just hit a real API and
return fresh data. This is what separates "agentic RAG" from plain RAG:
plain RAG only ever looks at a static, pre-indexed document store; an
agentic system can reach out to the live world when the static knowledge
base can't answer the question.

Both tools below use free, key-less public APIs so you can run this
project immediately without signing up for anything. In a real job,
you'd swap wikipedia_search for a paid search API (e.g. Tavily, Bing,
SerpAPI) for broader coverage.
"""

import requests


def wikipedia_search(query: str, sentences: int = 3) -> str:
    """
    Fetches a short, current summary of a topic from Wikipedia.
    Good for factual / entity questions the local doc store won't have.
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_"),
            timeout=8,
        )
        if resp.status_code != 200:
            return f"No live Wikipedia result found for '{query}'."
        data = resp.json()
        extract = data.get("extract", "")
        if not extract:
            return f"No summary available for '{query}'."
        # crude sentence trim so we don't flood the LLM context
        parts = extract.split(". ")
        return ". ".join(parts[:sentences]).strip() + ("." if not extract.endswith(".") else "")
    except requests.RequestException as e:
        return f"Live search failed: {e}"


def get_current_weather(city: str) -> str:
    """
    Fetches live current weather for a city using Open-Meteo (no API key required).
    First geocodes the city name to lat/lon, then fetches current conditions.
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=8,
        ).json()

        if not geo.get("results"):
            return f"Could not find location '{city}'."

        loc = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=8,
        ).json()

        cw = weather.get("current_weather", {})
        if not cw:
            return f"Weather data unavailable for {city}."

        return (f"Current weather in {loc.get('name', city)}: "
                f"{cw.get('temperature')}°C, wind {cw.get('windspeed')} km/h, "
                f"recorded at {cw.get('time')}.")
    except requests.RequestException as e:
        return f"Live weather fetch failed: {e}"


# Registry the agent uses to look up tools by name at runtime.
TOOL_REGISTRY = {
    "wikipedia_search": {
        "fn": wikipedia_search,
        "description": "Look up a current factual summary of a topic/entity from Wikipedia. "
                        "Input: a search term (e.g. 'Python programming language').",
    },
    "get_current_weather": {
        "fn": get_current_weather,
        "description": "Get real-time current weather for a named city. "
                        "Input: a city name (e.g. 'Chennai').",
    },
}
