import os
import json
import chromadb
from google.adk.agents import Agent
from sentence_transformers import SentenceTransformer
import typing
from typing import List, Dict, Any, Optional

# --- Configuration ---
# Match the settings used during indexing
CHROMA_DB_PATH = "./chroma_db_standard"
COLLECTION_NAME = "my_documents_collection_standard"
EMBEDDER_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Use the latest available flash model compatible with ADK
AGENT_MODEL_NAME = "gemini-1.5-flash"
# How many chunks to retrieve? Adjust based on context window needs and relevance
N_RETRIEVAL_RESULTS = 5

print("Initializing components...")


try:
    print(f"Loading embedding model: {EMBEDDER_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device="cuda") # Use 'cuda' if GPU available
    print("Embedding model loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit(1)

try:
    print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Connected to ChromaDB collection: {COLLECTION_NAME} (Count: {collection.count()})")
except Exception as e:
    # Common errors: Collection not found, path incorrect, DB corrupted
    print(f"Error connecting to ChromaDB collection '{COLLECTION_NAME}': {e}")
    print("Please ensure the database was created successfully and the path/name are correct.")
    exit(1)

print("Components initialized successfully.")


def retrieve_document_chunks_tool(query: str) -> Dict:
    """
    Retrieves relevant document chunks from the ChromaDB vector store.

    Args:
        query: The user's query text.

    Returns:
        A dictionary containing the retrieved chunks and status.
    """
    print(f"\n RAG Tool: Received query: '{query}'")
    if not query:
        print({"retrieved_chunks": [], "status": "error", "message": "Query cannot be empty."})
        return {"retrieved_chunks": [], "status": "error", "message": "Query cannot be empty."}
        

    try:
        # 1. Embed the query
        print(" RAG Tool: Embedding query...")
        query_embedding = embedder.encode([query])[0].tolist()
        print(" RAG Tool: Query embedded.")

        # 2. Query ChromaDB
        print(f" RAG Tool: Querying ChromaDB collection '{COLLECTION_NAME}'...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RETRIEVAL_RESULTS,
            include=['documents']
        )
        print(f" RAG Tool: ChromaDB query successful.")

        # 3. Extract results
        retrieved_docs = results.get('documents', [[]])[0] # Get the list of documents for the first query

        if not retrieved_docs:
            print(" RAG Tool: No relevant chunks found.")
            return {"retrieved_chunks": [], "status": "no_results", "message": "No relevant information found in the documents."}
        else:
            print(f" RAG Tool: Retrieved {len(retrieved_docs)} chunks.")
            return {"retrieved_chunks": retrieved_docs, "status": "success", "message": f"Retrieved {len(retrieved_docs)} relevant chunks."}

    except Exception as e:
        print(f" RAG Tool: Error during retrieval: {e}")
        return {"retrieved_chunks": [], "status": "error", "message": f"An error occurred during document retrieval: {e}"}

# --- Define the RAG Agent ---

print("Defining the RAG Agent...")

# Clear instructions are crucial for RAG agents
agent_instructions = f"""
You are a helpful assistant that answers user questions based on the information provided in the retrieved document chunks.

Your Task:
1.  Analyze the user's query.
2.  If the query requires searching the knowledge base, use the tool named 'retrieve_document_chunks_tool' to find relevant text chunks. <<< Make sure the tool name matches the Python function name
3.  Examine the dictionary returned by the 'retrieve_document_chunks_tool':
    - Look for the 'status' key. If status is 'success' and the 'retrieved_chunks' list is not empty: Carefully review the text in 'retrieved_chunks'. Synthesize an answer to the user's query *using only the information present in these chunks*. Do NOT add any information not found in the chunks.
    - If status is 'no_results' or 'error', or if the 'retrieved_chunks' list is empty: State clearly that you cannot answer the question based on the available documents.
4.  Do NOT use your general knowledge or search the web. Your knowledge is strictly limited to the documents retrieved by the tool.
5.  If the user query is not a question seeking information from the documents (e.g., a greeting, unrelated task), state that you are designed to answer questions based on the specific document set and cannot fulfill the request.
"""

# Create the Agent instance
root_agent  = Agent(
    model=AGENT_MODEL_NAME,
    tools=[retrieve_document_chunks_tool],
    instruction=agent_instructions,
    name="DocumentRAGAgent",
    description="Answers questions based on a specific set of indexed documents using RAG."
)

print("RAG Agent defined successfully.")