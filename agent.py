"""
agent.py
--------
No-cost Agentic RAG implementation.

The agent uses:
1. Local semantic search over documents
2. Local semantic search over conversation memory
3. Simple agentic action selection
4. Persistent episodic memory

No paid LLM API is required.
"""

import re

import config
from memory_store import VectorMemoryStore


class AgenticRAG:

    def __init__(self):
        self.memory = VectorMemoryStore()
        self.conversation_history = []

    # ---------------------------------------------------------
    # KNOWLEDGE SEARCH
    # ---------------------------------------------------------

    def _search_knowledge(self, query: str) -> str:

        results = self.memory.search(
            query,
            top_k=config.TOP_K_RETRIEVAL,
            source_filter="any"
        )

        if not results:
            return "No relevant information was found in the knowledge base."

        formatted = []

        for result in results:

            if result.source == "document":
                tag = "[Document]"
            else:
                tag = "[Past conversation]"

            formatted.append(
                f"{tag} {result.text}"
            )

        return "\n\n".join(formatted)

    # ---------------------------------------------------------
    # SIMPLE AGENT DECISION
    # ---------------------------------------------------------

    def _choose_action(self, query: str) -> str:

        query_lower = query.lower()

        knowledge_keywords = [
            "what",
            "why",
            "how",
            "explain",
            "define",
            "tell me",
            "project",
            "agentic",
            "rag",
            "vector",
            "embedding",
            "memory",
            "document",
            "conversation",
            "previous",
            "remember"
        ]

        if any(keyword in query_lower for keyword in knowledge_keywords):
            return "search_knowledge"

        # Default action for unknown/general questions
        return "search_knowledge"

    # ---------------------------------------------------------
    # GENERATE ANSWER FROM RETRIEVED INFORMATION
    # ---------------------------------------------------------

    def _generate_answer(self, question: str, context: str) -> str:

        if not context or context.startswith("No relevant"):
            return (
                "I couldn't find relevant information in my current "
                "knowledge base for that question."
            )

        # Extract individual retrieved pieces
        parts = context.split("\n\n")

        useful_parts = []

        for part in parts:

            cleaned = re.sub(
                r"^\[(Document|Past conversation)\]\s*",
                "",
                part
            )

            if cleaned.strip():
                useful_parts.append(cleaned.strip())

        if not useful_parts:
            return (
                "I couldn't find enough relevant information "
                "to answer that question."
            )

        # Avoid simply dumping too much information
        answer_parts = useful_parts[:3]

        answer = (
            "Based on the information available in my knowledge base:\n\n"
        )

        for part in answer_parts:
            answer += f"• {part}\n\n"

        return answer.strip()

    # ---------------------------------------------------------
    # MEMORY
    # ---------------------------------------------------------

    def _remember_exchange(self, question: str, answer: str):

        summary = (
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

        self.memory.add(
            summary,
            source="episodic",
            metadata={
                "type": "qa_pair"
            }
        )

        self.conversation_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

    # ---------------------------------------------------------
    # MAIN AGENT LOOP
    # ---------------------------------------------------------

    def ask(self, user_query: str, verbose: bool = True) -> str:

        if not user_query.strip():
            return "Please enter a question."

        seen_actions = set()

        for step in range(config.MAX_AGENT_STEPS):

            action = self._choose_action(user_query)

            if verbose:
                print(
                    f"\n[Step {step + 1}] "
                    f"Action: {action}"
                )

            # Prevent repeating exactly the same action forever
            action_key = (
                action,
                user_query.strip().lower()
            )

            if action_key in seen_actions:
                break

            seen_actions.add(action_key)

            # -------------------------------------------------
            # Execute action
            # -------------------------------------------------

            if action == "search_knowledge":

                observation = self._search_knowledge(
                    user_query
                )

            else:

                observation = (
                    "No action was available for this question."
                )

            if verbose:
                print(
                    f"[Step {step + 1}] "
                    f"Observation: {observation[:300]}..."
                )

            # -------------------------------------------------
            # Generate final answer
            # -------------------------------------------------

            answer = self._generate_answer(
                user_query,
                observation
            )

            self._remember_exchange(
                user_query,
                answer
            )

            return answer

        return (
            "I couldn't complete the reasoning process. "
            "Please try asking the question in another way."
        )