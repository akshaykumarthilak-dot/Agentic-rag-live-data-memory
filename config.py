"""
config.py
---------
Central place for all settings. Loads API keys from a .env file so you
never hardcode secrets in source code (this is itself an interview talking
point: "I keep secrets out of version control via environment variables").
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if present

# --- LLM settings ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 1024

# --- Embedding model (runs locally, no API key needed) ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good enough for demo/portfolio scale

# --- Storage paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store.pkl")
CONVERSATION_LOG_PATH = os.path.join(DATA_DIR, "conversation_log.jsonl")

# --- Agent behaviour ---
MAX_AGENT_STEPS = 5          # hard cap so the agent can't loop forever
TOP_K_RETRIEVAL = 4          # how many chunks/memories to pull per query
CHUNK_SIZE = 500             # characters per document chunk
CHUNK_OVERLAP = 50           # overlap between chunks so context isn't cut mid-sentence

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
