"""Command line: python -m agent "your question"   (or no argument for a chat loop)."""
import sys

from .graph import ask, format_answer

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(format_answer(ask(" ".join(sys.argv[1:]))))
    else:
        history = []
        print("Breast cancer data agent. Type 'exit' to quit.")
        while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
            out = ask(q, history)
            history.append({"question": q, "answer": out["answer"]})
            print("\nAgent:", format_answer(out))
