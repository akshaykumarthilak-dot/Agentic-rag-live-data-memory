# Agentic RAG Over Live Data & Memory

An AI assistant that combines three knowledge sources and decides, on its own,
which one(s) to use for a given question:

1. **Documents (RAG)** — your own `.txt` files, chunked and embedded.
2. **Live data** — real-time facts pulled from the web at query time (Wikipedia, weather).
3. **Memory** — a persistent log of past Q&A pairs, so the agent can recall earlier
   conversations across sessions (not just within one chat window).

Unlike a plain RAG chatbot, this system *reasons about which source to consult* rather
than always doing a fixed retrieve-then-answer pipeline. That reasoning loop is what
makes it "agentic."

---

## 1. Architecture

```
                     ┌─────────────────────┐
                     │      main.py         │  (CLI loop)
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │      agent.py         │  <-- the ReAct loop lives here
                     │  (AgenticRAG class)   │
                     └──┬────────┬────────┬──┘
                        │        │        │
          ┌─────────────▼──┐  ┌──▼─────┐ ┌▼──────────────┐
          │ memory_store.py │  │tools.py│ │ Anthropic API  │
          │ (vector search) │  │ (live  │ │ (reasoning /   │
          │                 │  │  data) │ │  final answer) │
          └─────────────────┘  └────────┘ └────────────────┘
```

- `rag_ingest.py` is run **once** (or whenever you add docs) to populate the vector store.
- `memory_store.py` holds BOTH document chunks and past conversation summaries in the
  same index — they're just tagged differently (`source="document"` vs `source="episodic"`).
- `tools.py` holds functions that hit real, live APIs — these are what make the agent
  "aware" of information that didn't exist when you built your document store.
- `agent.py` is the brain: it asks the LLM "what should I do next?", executes whatever
  it decides, and loops until it has enough to answer.

---

## 2. How the reasoning loop actually works (ReAct pattern)

Every step, the LLM is given:
- the original question
- everything it has learned so far (past "observations")

...and it must respond with **strict JSON** choosing ONE of:
- `search_knowledge` → look in your documents / past conversations
- `wikipedia_search` → look up a live fact
- `get_current_weather` → look up live weather
- `final_answer` → stop and answer the user

We parse that JSON in Python, run the requested action, and feed the *result* back
to the model as an "Observation" — then ask it to decide again. This repeats until
it picks `final_answer`, or we hit a safety cap (`MAX_AGENT_STEPS`, default 5) so a
confused model can't loop forever and burn API credits.

This is the same core pattern used by production agent frameworks (LangChain agents,
AutoGPT, etc.) — implemented here by hand so you can explain every line instead of
saying "the framework handles that."

---

## 3. Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

# 3. Build the knowledge base from docs/sample.txt (or add your own .txt files first)
python rag_ingest.py

# 4. Chat with the agent
python main.py
```

Try asking:
- `"What database does Project Nova use?"` → should trigger `search_knowledge` (it's in sample.txt)
- `"What's the weather in Chennai right now?"` → should trigger `get_current_weather`
- `"Who created Python?"` → should trigger `wikipedia_search`
- Ask something you asked earlier in a *previous run* → it should recall it from
  episodic memory, proving memory persists across sessions (it's saved to
  `data/vector_store.pkl` on disk, not just kept in RAM).

---

## 4. What to say about this project in an interview

- **Why not just use LangChain?** Building the ReAct loop by hand means you can
  explain exactly how action-selection, observation-injection, and loop termination
  work — a framework black-box doesn't give you that.
- **Why one vector store for both RAG and memory?** At query time, relevance is
  relevance — the agent doesn't need to know if a fact came from a file or a past
  chat. Tagging by `source` still lets you filter or weight them differently later.
- **What would you improve for production?** Swap the pickle-file store for a real
  vector DB (pgvector, Pinecone, Qdrant) for concurrency and scale; add a re-ranking
  step after retrieval; add streaming responses; move the loop-safety check from
  "exact repeat" to "semantic similarity" so the model can't rephrase its way around
  the repeat-guard.

---

## 5. Deploying it live on the web (Streamlit Community Cloud, free)

`streamlit_app.py` wraps the same `agent.py` in a chat UI. To make it public:

1. Push this whole folder to a **public GitHub repo** (a private repo also works once
   you connect your GitHub account to Streamlit).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", pick your repo, and set the main file path to `streamlit_app.py`.
4. Under the app's **Settings → Secrets**, add:
   ```
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
   (Never commit your real key to GitHub — `.env` should be in `.gitignore`.)
5. Click Deploy. You'll get a public URL like `your-app-name.streamlit.app`.

Notes:
- The vector store (`data/vector_store.pkl`) and any docs you ingested need to already
  be committed to the repo (or ingested via a startup script) — Streamlit Cloud's
  filesystem resets on redeploy, so anything created only at runtime won't persist
  between deploys (episodic memory added during a live session will persist for that
  running instance, but gets wiped on the next redeploy/restart).
- Free tier gives ~1GB RAM — `sentence-transformers` with the MiniLM model fits
  comfortably, but if you later swap to a bigger embedding model, watch memory usage.
- If you outgrow the free tier or want a "real" production feel (custom domain, no
  cold starts, a proper REST API instead of just a chat UI), the next step up is
  deploying a FastAPI backend on Render or Railway (both have free/cheap tiers) with
  a small separate frontend — happy to build that version too if you want it.

## 6. Extending this into the guardrail/eval project we discussed

If you want to build on this for the "evaluated guardrail agent + CI" project:
1. Write a small `evals/` folder with test cases: `{question, expected_action, forbidden_phrases}`.
2. After each agent response, check: did it call the *expected* action? Does the
   output contain anything from `forbidden_phrases` (e.g. leaked system prompt,
   PII, unsafe instructions)?
3. Wire that eval script into a GitHub Actions workflow (`.github/workflows/eval.yml`)
   that runs on every push and fails the build if any eval fails.

That turns this project from "a demo" into "a demo with automated regression testing,"
which is exactly the differentiator we talked about.
