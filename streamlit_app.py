import os
import streamlit as st

from agent import AgenticRAG
from rag_ingest import ingest_docs


st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="centered"
)


KNOWLEDGE_BASE_VERSION = "project-nova-v2"


def initialize_knowledge_base():

    data_dir = os.path.join(
        os.path.dirname(__file__),
        "data"
    )

    vector_store_path = os.path.join(
        data_dir,
        "vector_store.pkl"
    )

    version_path = os.path.join(
        data_dir,
        "knowledge_base_version.txt"
    )

    current_version = ""

    if os.path.exists(version_path):

        with open(
            version_path,
            "r",
            encoding="utf-8"
        ) as f:

            current_version = f.read().strip()

    # Rebuild the vector store if the knowledge-base
    # version has changed.
    if (
        not os.path.exists(vector_store_path)
        or current_version != KNOWLEDGE_BASE_VERSION
    ):

        if os.path.exists(vector_store_path):
            os.remove(vector_store_path)

        with st.spinner(
            "Building knowledge base..."
        ):

            ingest_docs()

        with open(
            version_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(KNOWLEDGE_BASE_VERSION)


# Initialize knowledge base
initialize_knowledge_base()

# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.title("🤖 Agentic RAG Assistant")

st.caption(
    "Answers using your documents and past conversations "
    "with local semantic retrieval."
)


# ---------------------------------------------------------
# AGENT
# ---------------------------------------------------------

if "agent" not in st.session_state:

    st.session_state.agent = AgenticRAG()


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


agent = st.session_state.agent


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.subheader("Knowledge base status")

    st.write(
        f"📄 Document chunks: "
        f"{agent.memory.count('document')}"
    )

    st.write(
        f"🧠 Past conversations: "
        f"{agent.memory.count('episodic')}"
    )

    st.caption(
        "Documents are embedded locally using "
        "sentence-transformers."
    )


# ---------------------------------------------------------
# DISPLAY PREVIOUS CHAT
# ---------------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

user_input = st.chat_input(
    "Ask me anything..."
)


if user_input:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # -----------------------------------------------------
    # AGENT RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            answer = agent.ask(
                user_input,
                verbose=False
            )

        st.markdown(answer)


    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )