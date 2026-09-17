"""
Terminal interface, matching the same pattern EIOS used (a terminal
interface alongside the MCP server and web demo). Useful for quick manual
testing without spinning up FastAPI.

Run with: PYTHONPATH=. python -m src.cli
"""

from __future__ import annotations

import json

from src.orchestrator import BiodiversityAssistant


def main() -> None:
    print("Darukaa.Earth Biodiversity Intelligence — terminal mode")
    print("Type your land's environmental details in plain text, or paste a")
    print("JSON object. Type 'exit' to quit.\n")

    assistant = BiodiversityAssistant()
    session_id = "cli-session"

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        structured = None
        text = user_input
        if user_input.startswith("{"):
            try:
                structured = json.loads(user_input)
                text = None
            except json.JSONDecodeError:
                pass

        result = assistant.handle_message(
            session_id=session_id, text=text, structured_input=structured
        )

        if result["type"] == "clarifying_question":
            print(f"\n[assistant] {result['question']}\n")
        else:
            print("\n" + json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
