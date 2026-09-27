"""
agent.py
--------
No-cost Agentic RAG implementation.

The agent uses:
1. Local semantic search over documents
2. Local semantic search over conversation memory
3. Live web search
4. Live weather data
5. Simple agentic action selection
6. Persistent episodic memory

No paid LLM API is required.
"""

import re
import uuid

import config
from memory_store import VectorMemoryStore
from tools import TOOL_REGISTRY


class AgenticRAG:

    def __init__(self, session_id: str = None):

        self.memory = VectorMemoryStore()

        # Unique ID for the current conversation
        self.session_id = (
            session_id
            if session_id
            else str(uuid.uuid4())
        )

        # Short-term memory
        # Stores recent messages from this session.
        self.conversation_history = []

        # Maximum number of recent exchanges
        # kept in short-term memory.
        self.max_history = 6

    # ---------------------------------------------------------
    # CONVERSATION CONTEXT
    # ---------------------------------------------------------

    def _build_contextual_query(self, user_query: str) -> str:

        if not self.conversation_history:
            return user_query

        recent_history = self.conversation_history[-3:]

        context_parts = []

        for item in recent_history:

            context_parts.append(
                f"Previous question: {item['question']}\n"
                f"Previous answer: {item['answer']}"
            )

        conversation_context = "\n\n".join(context_parts)

        return (
            f"Conversation context:\n"
            f"{conversation_context}\n\n"
            f"Current question: {user_query}"
        )

    # ---------------------------------------------------------
    # KNOWLEDGE SEARCH
    # ---------------------------------------------------------

    def _search_knowledge(self, query: str) -> str:

        # 1. Search documents first
        document_results = self.memory.search(
            query,
            top_k=config.TOP_K_RETRIEVAL,
            source_filter="document"
        )

        # If documents contain information, use them first
        if document_results:

            formatted = []

            for result in document_results:

                formatted.append(
                    f"[Document] {result.text}"
                )

            return "\n\n".join(formatted)

        # 2. If no document information exists,
        # search previous conversations
        memory_results = self.memory.search(
            query,
            top_k=config.TOP_K_RETRIEVAL,
            source_filter="episodic"
        )

        if not memory_results:

            return (
                "No relevant information was found "
                "in the knowledge base."
            )

        formatted = []

        for result in memory_results:

            formatted.append(
                f"[Past conversation] {result.text}"
            )

        return "\n\n".join(formatted)

    # ---------------------------------------------------------
    # LIVE WEB SEARCH
    # ---------------------------------------------------------

    def _search_live_web(self, query: str) -> str:

        tool = TOOL_REGISTRY["live_web_search"]

        return tool["fn"](query)

    # ---------------------------------------------------------
    # LIVE WEATHER
    # ---------------------------------------------------------

    def _search_weather(self, city: str) -> str:

        tool = TOOL_REGISTRY["get_current_weather"]

        return tool["fn"](city)

    # ---------------------------------------------------------
    # AGENT DECISION
    # ---------------------------------------------------------

    def _choose_action(self, query: str) -> str:

        query_lower = query.lower().strip()

        # -----------------------------------------------------
        # WEATHER QUESTIONS
        # -----------------------------------------------------

        weather_keywords = [
            "weather",
            "temperature",
            "rain",
            "raining",
            "forecast",
            "wind speed",
            "humidity"
        ]

        if any(
            keyword in query_lower
            for keyword in weather_keywords
        ):
            return "get_current_weather"

        # -----------------------------------------------------
        # LIVE / CURRENT INFORMATION
        # -----------------------------------------------------

        live_keywords = [
            "latest",
            "current",
            "today",
            "now",
            "recent",
            "recently",
            "news",
            "this week",
            "this month",
            "live",
            "price",
            "stock price",
            "release",
            "released",
            "new version",
            "latest version",
            "current version",
            "what happened"
        ]

        if any(
            keyword in query_lower
            for keyword in live_keywords
        ):
            return "live_web_search"

        # -----------------------------------------------------
        # KNOWLEDGE / DOCUMENT / MEMORY QUESTIONS
        # -----------------------------------------------------

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
            "remember",
            "backend",
            "database",
            "frontend",
            "authentication",
            "deployment",
            "jenkins",
            "docker"
        ]

        if any(
            keyword in query_lower
            for keyword in knowledge_keywords
        ):
            return "search_knowledge"

        # Default
        return "search_knowledge"

    # ---------------------------------------------------------
    # EXTRACT CITY FOR WEATHER
    # ---------------------------------------------------------

    def _extract_city(self, query: str) -> str:

        query_lower = query.lower()

        common_cities = [
            "chennai",
            "bangalore",
            "bengaluru",
            "mumbai",
            "delhi",
            "hyderabad",
            "kolkata",
            "pune",
            "coimbatore",
            "madurai",
            "tambaram"
        ]

        for city in common_cities:

            if city in query_lower:
                return city

        # Default city
        return "Chennai"

    # ---------------------------------------------------------
    # GENERATE ANSWER
    # ---------------------------------------------------------

    def _generate_answer(
        self,
        question: str,
        context: str,
        action: str
    ) -> str:

        if not context:

            return (
                "I couldn't find relevant information "
                "for that question."
            )

        if context.startswith(
            "No relevant information"
        ):

            return (
                "I couldn't find relevant information "
                "in my current knowledge base."
            )

        # -----------------------------------------------------
        # LIVE WEB ANSWER
        # -----------------------------------------------------

        if action == "live_web_search":

            return self._format_live_answer(
                context
            )

        # -----------------------------------------------------
        # WEATHER ANSWER
        # -----------------------------------------------------

        if action == "get_current_weather":

            return (
                "Based on the latest weather data:\n\n"
                f"• {context}"
            )

        # -----------------------------------------------------
        # DOCUMENT / MEMORY ANSWER
        # -----------------------------------------------------

        parts = context.split("\n\n")

        useful_parts = []

        for part in parts:

            cleaned = re.sub(
                r"^\[(Document|Past conversation)\]\s*",
                "",
                part
            )

            if cleaned.strip():

                useful_parts.append(
                    cleaned.strip()
                )

        if not useful_parts:

            return (
                "I couldn't find enough relevant "
                "information to answer that question."
            )

        answer_parts = useful_parts[:3]

        answer = (
            "Based on the information available "
            "in my knowledge base:\n\n"
        )

        for part in answer_parts:

            answer += f"• {part}\n\n"

        return answer.strip()

    # ---------------------------------------------------------
    # FORMAT LIVE WEB RESULTS
    # ---------------------------------------------------------

    def _format_live_answer(
        self,
        context: str
    ) -> str:

        results = context.split("\n\n")

        answer = (
            "Based on the latest information "
            "found on the web:\n\n"
        )

        sources = []

        for result in results[:3]:

            lines = result.split("\n")

            title = ""
            url = ""
            snippet = ""

            for line in lines:

                if line.startswith("Title:"):
                    title = line.replace(
                        "Title:",
                        "",
                        1
                    ).strip()

                elif line.startswith("URL:"):
                    url = line.replace(
                        "URL:",
                        "",
                        1
                    ).strip()

                elif line.startswith("Snippet:"):
                    snippet = line.replace(
                        "Snippet:",
                        "",
                        1
                    ).strip()

            if snippet:

                answer += (
                    f"• {snippet}\n\n"
                )

            if title and url:

                sources.append(
                    f"{len(sources) + 1}. "
                    f"[{title}]({url})"
                )

        if sources:

            answer += "### Sources\n\n"

            answer += "\n".join(
                sources
            )

        return answer.strip()

    # ---------------------------------------------------------
    # MEMORY
    # ---------------------------------------------------------
        # ---------------------------------------------------------
    # SHORT-TERM MEMORY
    # ---------------------------------------------------------

    def _get_recent_context(self) -> str:

        if not self.conversation_history:
            return ""

        recent = self.conversation_history[
            -self.max_history:
        ]

        context_parts = []

        for item in recent:

            context_parts.append(
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}"
            )

        return "\n\n".join(
            context_parts
        )
    def _remember_exchange(
        self,
        question: str,
        answer: str
    ):

        # -----------------------------------------------------
        # SHORT-TERM MEMORY
        # -----------------------------------------------------

        self.conversation_history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        # Keep only recent conversation
        if len(
            self.conversation_history
        ) > self.max_history:

            self.conversation_history = (
                self.conversation_history[
                    -self.max_history:
                ]
            )

        # -----------------------------------------------------
        # LONG-TERM EPISODIC MEMORY
        # -----------------------------------------------------

        summary = (
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

        self.memory.add(

            summary,

            source="episodic",

            metadata={
                "type": "qa_pair",
                "session_id": self.session_id
            },

            session_id=self.session_id,

            importance=0.6
        )
    # ---------------------------------------------------------
    # MAIN AGENT LOOP
    # ---------------------------------------------------------

    def ask(
        self,
        user_query: str,
        verbose: bool = True
    ) -> str:

        if not user_query.strip():

            return "Please enter a question."

        seen_actions = set()

        for step in range(
            config.MAX_AGENT_STEPS
        ):

            # -------------------------------------------------
            # Decide which tool/action to use
            # -------------------------------------------------

            action = self._choose_action(
                user_query
            )

            if verbose:

                print(
                    f"\n[Step {step + 1}] "
                    f"Action: {action}"
                )

            action_key = (
                action,
                user_query.strip().lower()
            )

            # Prevent repeating the same action
            if action_key in seen_actions:

                break

            seen_actions.add(
                action_key
            )
             # -------------------------------------------------
            # Retrieve short-term conversation context
            # -------------------------------------------------

            recent_context = (
                self._get_recent_context()
            )

            if verbose and recent_context:

                print(
                    "[Short-term memory] "
                    f"{recent_context[:300]}..."
                )
            # -------------------------------------------------
            # Execute action
            # -------------------------------------------------

            if action == "search_knowledge":

                contextual_query = self._build_contextual_query(
                    user_query
                )

                observation = self._search_knowledge(
                    contextual_query
                )

            elif action == "live_web_search":

                observation = self._search_live_web(
                    user_query
                )

            elif action == "get_current_weather":

                city = self._extract_city(
                    user_query
                )

                observation = self._search_weather(
                    city
                )

            else:

                observation = (
                    "No action was available "
                    "for this question."
                )

            if verbose:

                print(
                    f"[Step {step + 1}] "
                    f"Observation: "
                    f"{observation[:500]}..."
                )

            # -------------------------------------------------
            # Generate final answer
            # -------------------------------------------------

            answer = self._generate_answer(
                user_query,
                observation,
                action
            )

            # -------------------------------------------------
            # Remember interaction
            # -------------------------------------------------

            self._remember_exchange(
                user_query,
                answer
            )

            return answer

        return (
            "I couldn't complete the reasoning process. "
            "Please try asking the question in another way."
        )