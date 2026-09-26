"""
main.py
-------
Interactive CLI for the Agentic RAG assistant.

Setup:
    1. pip install -r requirements.txt
    2. Create a .env file with: ANTHROPIC_API_KEY=your_key_here
    3. Put some .txt files in docs/ (a sample.txt is included)
    4. python rag_ingest.py          # builds the document knowledge base
    5. python main.py                # start chatting
"""

from agent import AgenticRAG


def main():
    print("=" * 60)
    print("Agentic RAG Assistant (memory + live data + documents)")
    print("Type 'exit' to quit.")
    print("=" * 60)

    agent = AgenticRAG()
    print(f"\nLoaded memory: {agent.memory.count('document')} document chunks, "
          f"{agent.memory.count('episodic')} past conversation entries.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        answer = agent.ask(user_input, verbose=True)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
