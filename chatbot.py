"""
chatbot.py
----------
Command-line chat interface. Good for quick testing before you build
or demo the Streamlit UI.

Run:
    python chatbot.py
"""

from rag_chain import load_qa_chain, ask


def main():
    print("Loading vector store and model...")
    qa_chain = load_qa_chain()
    print("\nCollege Q&A Chatbot ready! Ask about syllabus, exam rules, "
          "placements, etc. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        answer, sources = ask(qa_chain, question)
        print(f"\nBot: {answer}\n")

        if sources:
            print("Sources:")
            for doc in sources:
                page = doc.metadata.get("page", "?")
                src = doc.metadata.get("source", "unknown file")
                print(f"  - {src} (page {int(page) + 1 if isinstance(page, int) else page})")
        print()


if __name__ == "__main__":
    main()
