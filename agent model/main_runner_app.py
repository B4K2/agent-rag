import asyncio
import uuid
from pathlib import Path

try:
    from google.adk.runners import Runner
    from google.genai import types as genai_types
except ImportError:
    print("Ensure google-adk and google-generativeai libraries are installed.")
    exit(1)

try:
    from agent import AGENT as root_agent_instance
    from agent import APP_NAME as RAG_APP_NAME
    from agent import _last_retrieved_chunks_for_see
    from session_memory import session_service, memory_service
except ImportError as e:
    print(f"Error importing from agent.py or session_memory.py: {e}")
    exit(1)

try:
    print(f"Initializing ADK Runner for app: {RAG_APP_NAME}...")
    universal_runner = Runner(
        agent=root_agent_instance,
        app_name=RAG_APP_NAME,
        session_service=session_service,
        memory_service=memory_service
    )
    print("ADK Runner initialized successfully.")
except Exception as e:
    print(f"Failed to initialize ADK Runner: {e}")
    exit(1)

async def process_user_query(
    runner: Runner,
    query: str,
    user_id: str,
    session_id: str,
    detailed_see: bool = False
) -> dict:
    global _last_retrieved_chunks_for_see
    _last_retrieved_chunks_for_see.clear()

    session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )

    new_user_message = genai_types.Content(role='user', parts=[genai_types.Part(text=query)])

    final_agent_response_text = "Error: Agent did not produce a final response."
    all_events_for_see = []
    tool_calls_summary = []

    print(f"Processing query for session {session_id[:8]}...")
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_user_message
    ):
        if detailed_see:
            try:
                all_events_for_see.append(event.to_dict() if hasattr(event, 'to_dict') else str(event))
            except Exception:
                all_events_for_see.append(f"Could not serialize event: {type(event)}")

        actions_obj = event.actions
        if actions_obj and hasattr(actions_obj, 'tool_code_execution') and actions_obj.tool_code_execution:
            for action in actions_obj.tool_code_execution.tool_actions:
                tool_calls_summary.append({
                    "tool_name": action.tool_name,
                    "status": "invoked"
                })
        elif actions_obj and hasattr(actions_obj, 'tool_calls') and actions_obj.tool_calls:
            for tool_call_action in actions_obj.tool_calls:
                if hasattr(tool_call_action, 'function_call'):
                    tool_calls_summary.append({
                        "tool_name": tool_call_action.function_call.name,
                        "status": "invoked"
                    })

        if event.is_final_response():
            if event.content and event.content.parts:
                final_agent_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_agent_response_text = f"Agent Escalated: {event.error_message or 'No reason provided.'}"
            break

    response_data = {
        "answer": final_agent_response_text,
        "session_id": session_id,
        "status": "success"
    }

    if detailed_see:
        response_data["retrieved_chunks"] = list(_last_retrieved_chunks_for_see)
        response_data["tool_calls_summary"] = tool_calls_summary

    return response_data

async def command_line_chat_loop():
    print("\n--- ADK Agent Command-Line Interface ---")
    print("Type 'new session' to start a new conversation.")
    print("Type 'quit' to exit.")

    current_user_id = "cli_user"
    current_session_id = str(uuid.uuid4())
    print(f"Starting new session: {current_session_id}")

    while True:
        user_input = input(f"\nYou ({current_session_id[:8]}): ")

        if user_input.lower() == 'quit':
            print("Exiting chat.")
            break
        if user_input.lower() == 'new session':
            current_session_id = str(uuid.uuid4())
            _last_retrieved_chunks_for_see.clear()
            print(f"\n--- Switched to new session: {current_session_id} ---")
            continue
        if not user_input.strip():
            continue

        show_details = False
        actual_query = user_input
        if user_input.lower().endswith("/see"):
            show_details = True
            actual_query = user_input[:-4].strip()
            print("  (Detailed 'see' mode requested)")

        results = await process_user_query(
            runner=universal_runner,
            query=actual_query,
            user_id=current_user_id,
            session_id=current_session_id,
            detailed_see=show_details
        )

        print(f"\nAgent ({current_session_id[:8]}): {results['answer']}")
        if show_details:
            if results.get("retrieved_chunks"):
                print("\n  --- Retrieved Chunks ---")
                for i, chunk in enumerate(results["retrieved_chunks"]):
                    print(f"    Chunk {i+1}: {chunk[:150]}...")
            if results.get("tool_calls_summary"):
                print("\n  --- Tool Calls Summary ---")
                for tc in results["tool_calls_summary"]:
                    print(f"    - Tool: {tc['tool_name']} ({tc['status']})")

if __name__ == "__main__":
    try:
        asyncio.run(command_line_chat_loop())
    except KeyboardInterrupt:
        print("\nExiting...")