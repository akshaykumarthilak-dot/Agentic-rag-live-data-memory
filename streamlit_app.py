"""
streamlit_app.py
-----------------
Web UI for the Agentic RAG assistant. Same agent.py brain, just wrapped
in a chat interface instead of a terminal loop.

Run locally with:
    streamlit run streamlit_app.py

For deployment, see the "Deploying" section in README.md — the short
version is: push this repo to GitHub, connect it on
https://share.streamlit.io, set ANTHROPIC_API_KEY as a secret there,
and Streamlit gives you a public URL.
"""

import os
import streamlit as st

from agent import AgenticRAG

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🧠", layout="centered")

# --- API key handling: works both locally (.env) and on Streamlit Cloud (st.secrets) ---
if "ANTHROPIC_API_KEY" not in os.environ:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

st.title("🧠 Agentic RAG Assistant")
st.caption("Answers using your documents, past conversations, and live web data — the agent decides which.")

# --- one agent instance per user session, persisted memory on disk across runs ---
if "agent" not in st.session_state:
    st.session_state.agent = AgenticRAG()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = st.session_state.agent

with st.sidebar:
    st.subheader("Knowledge base status")
    st.write(f"📄 Document chunks: {agent.memory.count('document')}")
    st.write(f"💬 Past conversations: {agent.memory.count('episodic')}")
    st.caption("Run `python rag_ingest.py` locally and redeploy to add more documents.")

# --- render existing chat ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- handle new input ---
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = agent.ask(user_input, verbose=False)
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
