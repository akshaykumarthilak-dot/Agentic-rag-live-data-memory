"""
agent.py
--------
The "agentic" half of Agentic RAG.

This is a ReAct-style loop (Reason -> Act -> Observe -> repeat):

  1. The LLM sees the user's question + a list of available actions.
  2. It picks ONE action and returns it as JSON.
  3. We execute that action in Python (search memory, call a live tool, etc).
  4. We feed the result back to the LLM as an "observation".
  5. Repeat until the LLM is confident enough to give a final_answer,
     or we hit MAX_AGENT_STEPS (a safety cap against infinite loops).

This is deliberately implemented by hand (no LangChain/LlamaIndex) so that
every moving part is visible and you can explain each one line-by-line in
an interview - that transparency is worth more to a technical panel than
a framework you can't fully account for.
"""

import json
import re

from anthropic import Anthropic

import config
from memory_store import VectorMemoryStore
from tools import TOOL_REGISTRY

SYSTEM_PROMPT = """You are an autonomous research assistant with three ways to gather information
before answering:

1. search_knowledge - search the user's private document + conversation memory (RAG).
   Use this FIRST for anything that might be in the user's own uploaded documents
   or something discussed earlier in this conversation.
2. wikipedia_search - look up a live, current factual summary of a topic/entity.
3. get_current_weather - get live current weather for a named city.

On every turn, respond with ONLY a JSON object, no other text, in one of these two shapes:

To take an action:
{"thought": "<why you're doing this>", "action": "<search_knowledge|wikipedia_search|get_current_weather>", "action_input": "<string input for the action>"}

To answer the user once you have enough information:
{"thought": "<why you're confident now>", "action": "final_answer", "action_input": "<the full answer to give the user>"}

Rules:
- Always try search_knowledge at least once before concluding you need live data,
  UNLESS the question is clearly about something current/real-time (weather, news, "latest").
- Never call the same action with the same input twice.
- If tools return nothing useful after 2-3 tries, give the best answer you can and say
  what you couldn't confirm.
"""


def _extract_json(raw: str) -> dict:
    """LLMs sometimes wrap JSON in markdown fences or add stray text - strip that defensively."""
    raw = raw.strip()
    fenced = re.search(r"\{.*\}", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(0)
    return json.loads(raw)


class AgenticRAG:
    def __init__(self):
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.memory = VectorMemoryStore()
        self.conversation_history = []  # short-term buffer for THIS session only

    # ---------- action implementations ----------

    def _search_knowledge(self, query: str) -> str:
        results = self.memory.search(query, top_k=config.TOP_K_RETRIEVAL, source_filter="any")
        if not results:
            return "No relevant information found in documents or past conversations."
        formatted = []
        for r in results:
            tag = "[Document]" if r.source == "document" else "[Past conversation]"
            formatted.append(f"{tag} {r.text}")
        return "\n---\n".join(formatted)

    def _execute_action(self, action: str, action_input: str) -> str:
        if action == "search_knowledge":
            return self._search_knowledge(action_input)
        if action in TOOL_REGISTRY:
            return TOOL_REGISTRY[action]["fn"](action_input)
        return f"Unknown action '{action}'."

    # ---------- main loop ----------

    def ask(self, user_query: str, verbose: bool = True) -> str:
        messages = [
            {"role": "user", "content": f"User question: {user_query}"}
        ]
        seen_actions = set()

        for step in range(config.MAX_AGENT_STEPS):
            response = self.client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            raw_text = response.content[0].text

            try:
                parsed = _extract_json(raw_text)
            except (json.JSONDecodeError, IndexError):
                # Model didn't follow the format - treat its raw text as the final answer
                # rather than crashing the whole agent.
                return raw_text

            action = parsed.get("action")
            action_input = parsed.get("action_input", "")

            if verbose:
                print(f"\n[Step {step + 1}] Thought: {parsed.get('thought', '')}")
                print(f"[Step {step + 1}] Action: {action} -> {action_input!r}")

            if action == "final_answer":
                self._remember_exchange(user_query, action_input)
                return action_input

            action_key = (action, action_input)
            if action_key in seen_actions:
                # Break potential infinite loops where the model repeats itself
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user",
                                  "content": "Observation: You already tried this exact action. "
                                             "Please give a final_answer with what you know."})
                seen_actions.add(action_key)
                continue
            seen_actions.add(action_key)

            observation = self._execute_action(action, action_input)
            if verbose:
                print(f"[Step {step + 1}] Observation: {observation[:200]}...")

            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        # Safety net if we exhaust MAX_AGENT_STEPS without a final_answer
        return "I wasn't able to reach a confident answer within my step limit. Here's what I found:\n" + \
               "\n".join(m["content"] for m in messages if m["role"] == "user")[-800:]

    def _remember_exchange(self, question: str, answer: str):
        """
        This is the 'memory' feedback loop: after answering, store a compact
        summary of the exchange as an episodic memory so future questions
        (even in a later session, since this is persisted to disk) can
        reference what was previously discussed.
        """
        summary = f"Q: {question}\nA: {answer}"
        self.memory.add(summary, source="episodic", metadata={"type": "qa_pair"})
        self.conversation_history.append({"question": question, "answer": answer})
