# RAG Agent: PDF-Powered Conversational AI

This repository implements a Retrieval-Augmented Generation (RAG) agent using Google's Agent Development Kit (ADK), Gemini embeddings, ChromaDB, and PDF ingestion via the Marker CLI. It enables users to upload and process PDFs, transforming them into a searchable knowledge base used by a conversational agent.

---

## 🚀 Features

- **Conversational AI Agent** using Google ADK with Gemini LLM.
- **PDF Ingestion** via Marker CLI into structured Markdown format.
- **Vector Store** using ChromaDB with Gemini-generated embeddings.
- **Streamlit UI** for drag-and-drop PDF uploading and ingestion.
- **FastAPI Backend** for PDF ingestion and agent query endpoints.
- **Standalone CLI ingestion** for batch processing documents.

---

## 🧠 How It Works

1. **PDFs are uploaded** via Streamlit or FastAPI.
2. **Marker CLI** converts PDFs to Markdown.
3. **Text is chunked** and embedded using Gemini.
4. **Chunks are stored** in ChromaDB as a vector database.
5. **User queries** are embedded and matched to relevant chunks.
6. **Google ADK agent** generates a response using retrieved data.

---

## 🗂️ Project Structure

```
RAG_AGENT/
├── agent_model/              # ADK logic and memory services
├── chroma_db_gemini/         # ChromaDB vector store directory
├── OUTPUT/                   # Markdown outputs for ingestion
├── api_temp_uploads/         # FastAPI temporary PDF storage
├── api_marker_outputs/       # FastAPI Marker outputs
├── temp_uploads_cli/         # Streamlit temporary PDF storage
├── marker_cli_outputs/       # Streamlit Marker outputs
├── app.py                    # Streamlit app
├── main_api.py               # FastAPI backend
├── vector_db.py              # Chroma ingestion logic
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── README.md                 # Project documentation
```

---

## ⚙️ Installation

```bash
git clone https://github.com/B4K2/agent-rag.git
cd agent-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔧 Configuration

Create a `.env` file:

```env
GEMINI_API_KEY="your_google_gemini_api_key"
```

Install Marker CLI: https://github.com/VikParuchuri/marker

---

## 🧪 Running the Project

### 1. Standalone Ingestion

```bash
python vector_db.py
```

### 2. Streamlit UI

```bash
streamlit run app.py
```

### 3. FastAPI Backend

```bash
uvicorn main_api:app --reload --port 8000
```

### 4. ADK CLI Agent

```bash
adk dev "agent_model.agent"
```

---

## 🛠️ APIs

- `/run`: Ask a question to the agent
- `/run_see`: Same as `/run`, with retrieved chunks shown
- `/upload_and_ingest_pdfs`: Upload PDFs and ingest
- `/ingest_standalone_output`: Ingest from `./OUTPUT`

---

## 📌 Dependencies (requirements.txt)

```
streamlit
fastapi
uvicorn[standard]
python-multipart
google-generativeai>=0.5.0
google-adk
langchain
chromadb
python-dotenv
requests
beautifulsoup4
tqdm
```

---

## 📈 Future Enhancements

- Persistent memory (e.g. Firestore)
- Better error handling and logging
- Support for more tools (weather, time, web search)
- Replace in-memory RAG chunk store
- Security and auth for production endpoints

---

## 🧰 Troubleshooting

- **Marker not found?** Ensure it's installed and in PATH.
- **Chroma errors?** Run ingestion script and check paths.
- **Import issues?** Run from root directory and check `__init__.py`.
- **API key errors?** Check `.env` formatting and location.

---

## 📄 License

This project is under the MIT License. Feel free to use and adapt.

---
