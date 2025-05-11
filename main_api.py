import os
import json
import shutil
from pathlib import Path
import subprocess
import time
import uuid
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

import sys
CURRENT_SCRIPT_DIR = Path(__file__).parent
AGENT_MODEL_SUBDIR_NAME = "agent model"
AGENT_MODEL_FULL_PATH = CURRENT_SCRIPT_DIR / AGENT_MODEL_SUBDIR_NAME

if not AGENT_MODEL_FULL_PATH.is_dir():
    print(f"CRITICAL: Subdirectory '{AGENT_MODEL_SUBDIR_NAME}' not found at {AGENT_MODEL_FULL_PATH}")
    exit(1)

sys.path.insert(0, str(AGENT_MODEL_FULL_PATH.resolve()))
sys.path.insert(0, str(CURRENT_SCRIPT_DIR.resolve()))

try:
    from vector_db import (
        ingest_marker_output_to_chroma_gemini,
        DEFAULT_CHROMA_DB_PATH_GEMINI,
        DEFAULT_COLLECTION_NAME_GEMINI,
        DEFAULT_GEMINI_EMBEDDER_MODEL_ID,
        OUTPUT_FOLDER_PATH as STANDALONE_VECTOR_DB_INPUT_PATH
    )
    from agent import (
        AGENT as root_agent_instance,
        APP_NAME as RAG_APP_NAME,
        _last_retrieved_chunks_for_see
    )
    from session_memory import session_service, memory_service
    from google.adk.runners import Runner
    from google.genai import types as genai_types
except ImportError as e:
    print(f"CRITICAL: Failed to import necessary project modules: {e}")
    exit(1)

API_UPLOAD_FOLDER = Path("./api_temp_uploads")
API_MARKER_OUTPUT_BASE_DIR = Path("./api_marker_outputs")

API_UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
API_MARKER_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="RAG Agent API",
    description="API for interacting with the RAG agent, ingesting documents, and managing data."
)

try:
    print(f"FastAPI: Initializing ADK Runner for app: {RAG_APP_NAME}...")
    universal_runner_instance = Runner(
        agent=root_agent_instance,
        app_name=RAG_APP_NAME,
        session_service=session_service,
        memory_service=memory_service
    )
    print("FastAPI: ADK Runner initialized successfully.")
except Exception as e:
    print(f"FastAPI: CRITICAL - Failed to initialize ADK Runner: {e}")
    universal_runner_instance = None

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "api_user"

class AgentResponse(BaseModel):
    answer: str
    session_id: str

class AgentResponseDetailed(AgentResponse):
    retrieved_chunks: Optional[List[str]] = None
    tool_calls_summary: Optional[List[Dict[str, str]]] = None

class IngestionStatus(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class UploadAndIngestResponse(BaseModel):
    message: str
    processed_files_count: int
    marker_errors: List[Dict[str, Any]]
    ingestion_status: Optional[IngestionStatus] = None

class StandaloneIngestionRequest(BaseModel):
    source_directory: str = str(STANDALONE_VECTOR_DB_INPUT_PATH)
    chroma_db_path: str = DEFAULT_CHROMA_DB_PATH_GEMINI
    collection_name: str = DEFAULT_COLLECTION_NAME_GEMINI
    gemini_model_id: str = DEFAULT_GEMINI_EMBEDDER_MODEL_ID
    delete_collection: bool = False

async def _process_agent_query_internal(
    runner: Runner,
    query: str,
    user_id: str,
    session_id: str,
    detailed_see: bool = False
) -> dict:
    global _last_retrieved_chunks_for_see
    if runner is None:
        raise HTTPException(status_code=503, detail="ADK Runner is not available.")

    _last_retrieved_chunks_for_see.clear()

    session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id
    )
    new_user_message = genai_types.Content(role='user', parts=[genai_types.Part(text=query)])

    final_answer = "Error: Agent did not produce a final response."
    tool_calls = []

    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=new_user_message
    ):
        actions_obj = event.actions
        if actions_obj and hasattr(actions_obj, 'tool_code_execution') and actions_obj.tool_code_execution:
            for action in actions_obj.tool_code_execution.tool_actions:
                tool_calls.append({"tool_name": action.tool_name, "status": "invoked"})

        if event.is_final_response():
            if event.content and event.content.parts:
                final_answer = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_answer = f"Agent Escalated: {event.error_message or 'No reason provided.'}"
            break

    response_payload = {
        "answer": final_answer,
        "session_id": session_id
    }
    if detailed_see:
        response_payload["retrieved_chunks"] = list(_last_retrieved_chunks_for_see)
        response_payload["tool_calls_summary"] = tool_calls
    return response_payload

@app.post("/run", response_model=AgentResponse)
async def run_agent_query(request_data: QueryRequest):
    session_id_to_use = request_data.session_id or str(uuid.uuid4())
    try:
        result = await _process_agent_query_internal(
            runner=universal_runner_instance,
            query=request_data.query,
            user_id=request_data.user_id,
            session_id=session_id_to_use,
            detailed_see=False
        )
        return AgentResponse(**result)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in /run endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing agent query: {str(e)}")

@app.post("/run_see", response_model=AgentResponseDetailed)
async def run_agent_query_detailed(request_data: QueryRequest):
    session_id_to_use = request_data.session_id or str(uuid.uuid4())
    try:
        result = await _process_agent_query_internal(
            runner=universal_runner_instance,
            query=request_data.query,
            user_id=request_data.user_id,
            session_id=session_id_to_use,
            detailed_see=True
        )
        return AgentResponseDetailed(**result)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error in /run_see endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing detailed agent query: {str(e)}")

@app.post("/upload_and_ingest_pdfs", response_model=UploadAndIngestResponse)
async def upload_and_ingest_pdfs_api(
    files: List[UploadFile] = File(...),
    chroma_db_path: str = Query(DEFAULT_CHROMA_DB_PATH_GEMINI),
    collection_name: str = Query(DEFAULT_COLLECTION_NAME_GEMINI),
    gemini_model_id: str = Query(DEFAULT_GEMINI_EMBEDDER_MODEL_ID),
    delete_collection: bool = Query(False)
):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF files provided.")

    if API_MARKER_OUTPUT_BASE_DIR.exists():
        shutil.rmtree(API_MARKER_OUTPUT_BASE_DIR)
    API_MARKER_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    if API_UPLOAD_FOLDER.exists():
        shutil.rmtree(API_UPLOAD_FOLDER)
    API_UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    processed_files_count = 0
    marker_errors_list = []

    for file in files:
        if file.content_type != "application/pdf":
            marker_errors_list.append({"filename": file.filename, "error": "Invalid file type, must be PDF."})
            await file.close()
            continue

        pdf_name_stem = Path(file.filename).stem
        temp_pdf_path = API_UPLOAD_FOLDER / file.filename

        try:
            with open(temp_pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            marker_command = [
                "marker_single",
                str(temp_pdf_path.resolve()),
                "--output_dir",
                str(API_MARKER_OUTPUT_BASE_DIR.resolve())
            ]
            process = await asyncio.to_thread(
                subprocess.run, marker_command, capture_output=True, text=True, encoding='utf-8', check=False
            )

            if process.returncode == 0:
                expected_md_path = API_MARKER_OUTPUT_BASE_DIR / pdf_name_stem / f"{pdf_name_stem}.md"
                if expected_md_path.exists():
                    meta_data = {
                        "original_filename": file.filename,
                        "processed_by": "fastapi_marker_single_v2",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "api_source": "upload_endpoint"
                    }
                    meta_json_path = API_MARKER_OUTPUT_BASE_DIR / pdf_name_stem / f"{pdf_name_stem}_meta.json"
                    with open(meta_json_path, "w", encoding="utf-8") as mjf:
                        json.dump(meta_data, mjf, indent=2)
                    processed_files_count += 1
                else:
                    marker_errors_list.append({"filename": file.filename, "error": "Marker MD output not found.", "details": process.stderr or process.stdout})
            else:
                marker_errors_list.append({"filename": file.filename, "error": f"Marker failed (exit code {process.returncode}).", "details": process.stderr or process.stdout})

        except Exception as e_marker:
            marker_errors_list.append({"filename": file.filename, "error": f"Exception during Marker processing: {str(e_marker)}"})
        finally:
            if temp_pdf_path.exists():
                os.remove(temp_pdf_path)
            await file.close()

    ingestion_status_obj = None
    if processed_files_count > 0:
        try:
            ingestion_report = ingest_marker_output_to_chroma_gemini(
                marker_output_base_dir=str(API_MARKER_OUTPUT_BASE_DIR),
                chroma_db_path=chroma_db_path,
                collection_name=collection_name,
                gemini_embedder_model_id=gemini_model_id,
                delete_existing_collection=delete_collection
            )
            ingestion_status_obj = IngestionStatus(
                status=ingestion_report.get("status", "unknown_ingestion_status"),
                message=ingestion_report.get("message", "Ingestion outcome unclear."),
                details=ingestion_report
            )
        except Exception as e_ingest:
            ingestion_status_obj = IngestionStatus(status="ingestion_call_exception", message=str(e_ingest))
    elif not marker_errors_list:
        ingestion_status_obj = IngestionStatus(status="no_files_to_process", message="No files were available or valid for Marker processing.")

    return UploadAndIngestResponse(
        message=f"Marker processing attempted. Successful: {processed_files_count}. Errors: {len(marker_errors_list)}.",
        processed_files_count=processed_files_count,
        marker_errors=marker_errors_list,
        ingestion_status=ingestion_status_obj
    )

@app.post("/ingest_standalone_output", response_model=IngestionStatus)
async def ingest_standalone_output_folder(request_data: StandaloneIngestionRequest):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured for server.")

        ingestion_report = await asyncio.to_thread(
            ingest_marker_output_to_chroma_gemini,
            marker_output_base_dir=request_data.source_directory,
            chroma_db_path=request_data.chroma_db_path,
            collection_name=request_data.collection_name,
            gemini_embedder_model_id=request_data.gemini_model_id,
            delete_existing_collection=request_data.delete_collection,
            gemini_api_key_override=api_key
        )
        return IngestionStatus(
            status=ingestion_report.get("status", "unknown_ingestion_status"),
            message=ingestion_report.get("message", "Ingestion outcome unclear."),
            details=ingestion_report
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Standalone ingestion failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "runner_status": "available" if universal_runner_instance else "unavailable"}