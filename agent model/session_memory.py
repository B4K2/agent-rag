from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService

APP_NAME = "my_rag_app" 

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

print(f"Initialized session and memory services for app: {APP_NAME}")