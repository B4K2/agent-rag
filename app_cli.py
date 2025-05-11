# app.py (your Streamlit app)

import streamlit as st
import os
import json
import shutil
from pathlib import Path
import subprocess
import time

from vector_db import (
    ingest_marker_output_to_chroma_gemini,
    DEFAULT_CHROMA_DB_PATH_GEMINI,
    DEFAULT_COLLECTION_NAME_GEMINI,
    DEFAULT_GEMINI_EMBEDDER_MODEL_ID
)

UPLOAD_FOLDER = Path("./temp_uploads_cli")
MARKER_CLI_OUTPUT_BASE_DIR = Path("./marker_cli_outputs") # Where Marker saves outputs

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MARKER_CLI_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_folders():
    """Removes temporary upload and marker output folders."""
    if UPLOAD_FOLDER.exists():
        shutil.rmtree(UPLOAD_FOLDER)
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    
    if MARKER_CLI_OUTPUT_BASE_DIR.exists():
        shutil.rmtree(MARKER_CLI_OUTPUT_BASE_DIR)
    MARKER_CLI_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    st.success("Cleaned up temporary folders.")
    # Clear session state related to processing to reset UI
    if 'successful_marker_processing' in st.session_state:
        del st.session_state['successful_marker_processing']


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="PDF to ChromaDB (Gemini)")
st.title("📄 PDF to ChromaDB with Gemini Embeddings")
st.subheader("Uses `marker_single` CLI for PDF processing, then Gemini for vectorization.")

st.sidebar.header("Settings & Controls")

# Allow user to specify Gemini Model ID (optional)
st.sidebar.markdown("### Gemini Settings")
gemini_model_id_input = st.sidebar.text_input(
    "Gemini Embedding Model ID",
    value=DEFAULT_GEMINI_EMBEDDER_MODEL_ID, # Use the default from processor
    help="E.g., 'embedding-001', 'text-embedding-004'. Ensure this model was used if appending to existing DB."
)
# Allow user to specify ChromaDB path and collection (optional)
st.sidebar.markdown("### ChromaDB Settings")
chroma_db_path_input = st.sidebar.text_input(
    "ChromaDB Path",
    value=DEFAULT_CHROMA_DB_PATH_GEMINI # Use the default from processor
)
collection_name_input = st.sidebar.text_input(
    "ChromaDB Collection Name",
    value=DEFAULT_COLLECTION_NAME_GEMINI # Use the default from processor
)
delete_existing_collection_on_ingest = st.sidebar.checkbox(
    "Delete existing collection before new ingestion",
    value=False,
    help="If checked, the entire collection will be wiped before adding new documents from this batch."
)


if st.sidebar.button("🧹 Clean Temp Folders & Reset", key="clean_cli"):
    cleanup_folders()
    st.rerun()

# File Uploader
uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader_cli"
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} PDF(s):")
    for uf_display in uploaded_files:
        st.markdown(f"- `{uf_display.name}`")
    
    processing_placeholder = st.empty()

    if st.button("🚀 Process PDFs with `marker_single` CLI", key="process_pdfs_cli"):
        # Clear previous marker output for a clean run for this batch
        if MARKER_CLI_OUTPUT_BASE_DIR.exists():
            shutil.rmtree(MARKER_CLI_OUTPUT_BASE_DIR)
        MARKER_CLI_OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        st.session_state['successful_marker_processing'] = False # Reset flag

        processing_placeholder.info("Starting PDF processing with `marker_single` CLI... This can take some time per file.")
        
        progress_bar = st.progress(0.0, text="Overall PDF Processing Progress")
        status_texts_area = st.container() # Area for individual file statuses

        processed_marker_count = 0
        error_files_marker = []
        
        total_files_to_process = len(uploaded_files)
        for i, uploaded_file_obj in enumerate(uploaded_files):
            pdf_name_stem = Path(uploaded_file_obj.name).stem
            
            with status_texts_area: # Show status within this container
                st.info(f"({i+1}/{total_files_to_process}) 🔄 Processing with Marker: **{uploaded_file_obj.name}**...")

            temp_pdf_path = UPLOAD_FOLDER / uploaded_file_obj.name
            with open(temp_pdf_path, "wb") as f_pdf:
                f_pdf.write(uploaded_file_obj.getbuffer())
            
            command = [
                "marker_single",
                str(temp_pdf_path.resolve()),
                "--output_dir",  # Or whatever the correct flag is
                str(MARKER_CLI_OUTPUT_BASE_DIR.resolve()),
            ]
            
            log_output_marker = ""
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1, universal_newlines=True)
                
                stdout_lines_marker = []
                stderr_lines_marker = []

                # Stream output
                for stdout_line in process.stdout:
                    stdout_lines_marker.append(stdout_line.strip())
                for stderr_line in process.stderr: # Capture stderr separately
                    stderr_lines_marker.append(stderr_line.strip())
                
                process.wait() # Wait for the process to complete
                return_code_marker = process.returncode
                
                stdout_full_marker = "\n".join(stdout_lines_marker)
                stderr_full_marker = "\n".join(stderr_lines_marker)
                log_output_marker = f"MARKER STDOUT:\n{stdout_full_marker}\n\nMARKER STDERR:\n{stderr_full_marker}"

                if return_code_marker == 0:
                    # Marker output folder is MARKER_CLI_OUTPUT_BASE_DIR / pdf_name_stem
                    # MD file is MARKER_CLI_OUTPUT_BASE_DIR / pdf_name_stem / pdf_name_stem.md
                    expected_md_path = MARKER_CLI_OUTPUT_BASE_DIR / pdf_name_stem / f"{pdf_name_stem}.md"
                    if expected_md_path.exists():
                        with status_texts_area:
                            st.success(f"({i+1}/{total_files_to_process}) ✓ Marker processed: {uploaded_file_obj.name}")
                        
                        # Create the _meta.json file
                        meta_data_for_file = {
                            "original_filename": uploaded_file_obj.name,
                            "processed_by": "marker_single_cli_streamlit",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "marker_output_path": str(MARKER_CLI_OUTPUT_BASE_DIR / pdf_name_stem)
                        }
                        # Save _meta.json inside the marker output subfolder
                        meta_json_path = MARKER_CLI_OUTPUT_BASE_DIR / pdf_name_stem / f"{pdf_name_stem}_meta.json"
                        with open(meta_json_path, "w", encoding="utf-8") as mjf:
                            json.dump(meta_data_for_file, mjf, indent=2)
                        
                        processed_marker_count += 1
                    else:
                        with status_texts_area:
                            st.error(f"({i+1}/{total_files_to_process}) ✗ Error (Marker): {uploaded_file_obj.name}. Output MD not found.")
                        error_files_marker.append({"name": uploaded_file_obj.name, "log": log_output_marker, "reason": "Output MD not found by app"})
                else:
                    with status_texts_area:
                        st.error(f"({i+1}/{total_files_to_process}) ✗ Error (Marker): {uploaded_file_obj.name}. Exit code: {return_code_marker}")
                    error_files_marker.append({"name": uploaded_file_obj.name, "log": log_output_marker, "exit_code": return_code_marker})

            except FileNotFoundError:
                with status_texts_area:
                    st.error(f"({i+1}/{total_files_to_process}) ✗ CRITICAL: `marker_single` command not found. Is Marker installed and in your PATH?")
                error_files_marker.append({"name": uploaded_file_obj.name, "log": "marker_single not found by subprocess"})
                processing_placeholder.error("`marker_single` command not found. Aborting Marker processing.")
                if temp_pdf_path.exists(): os.remove(temp_pdf_path)
                break 
            except Exception as e_marker:
                with status_texts_area:
                    st.error(f"({i+1}/{total_files_to_process}) ✗ Error processing {uploaded_file_obj.name} with Marker: {str(e_marker)[:200]}...")
                error_files_marker.append({"name": uploaded_file_obj.name, "log": str(e_marker)})
            finally:
                if temp_pdf_path.exists():
                    os.remove(temp_pdf_path)
            
            progress_bar.progress((i + 1) / total_files_to_process, text=f"Overall PDF Processing Progress ({i+1}/{total_files_to_process})")

        processing_placeholder.empty()
        st.info(f"Marker CLI processing complete. Successfully processed: {processed_marker_count}, Errors: {len(error_files_marker)}.")
        
        if error_files_marker:
            st.error("Details for files with Marker processing errors:")
            for err_file_marker in error_files_marker:
                with st.expander(f"Marker Error: {err_file_marker['name']}"):
                    st.text(err_file_marker.get("log", "No log available."))

        if processed_marker_count > 0:
            st.session_state['successful_marker_processing'] = True
        else:
            st.session_state['successful_marker_processing'] = False


    # --- ChromaDB Ingestion Section ---
    st.divider()
    st.subheader("📦 Add to ChromaDB Vector Store (using Gemini)")
    st.caption(f"This will process files from: `{MARKER_CLI_OUTPUT_BASE_DIR.resolve()}`")
    st.caption(f"Database path: `{Path(chroma_db_path_input).resolve()}` | Collection: `{collection_name_input}` | Model: `{gemini_model_id_input}`")

    ready_for_chroma_ingestion = st.session_state.get('successful_marker_processing', False)

    if not ready_for_chroma_ingestion:
        if not uploaded_files:
            st.info("1. Upload PDF files.")
        else: 
            st.info("1. Upload PDF files.\n2. Process them successfully with `marker_single` CLI.")
    else: 
        if st.button("➕ Add Processed Files to ChromaDB (Gemini)", key="add_to_chroma_gemini"):
            with st.spinner(f"Ingesting documents into ChromaDB using Gemini ('{gemini_model_id_input}')... This may take a while."):
                try:
                    # Call the imported function from vector_db_processor.py
                    ingestion_stats = ingest_marker_output_to_chroma_gemini(
                        marker_output_base_dir=str(MARKER_CLI_OUTPUT_BASE_DIR),
                        chroma_db_path=chroma_db_path_input, # From sidebar
                        collection_name=collection_name_input, # From sidebar
                        gemini_embedder_model_id=gemini_model_id_input, # From sidebar
                        delete_existing_collection=delete_existing_collection_on_ingest # From sidebar
                    )
                    if ingestion_stats.get("status") == "success":
                        st.success("ChromaDB ingestion process finished successfully!")
                    elif ingestion_stats.get("status") == "warning":
                        st.warning(f"ChromaDB ingestion process completed with a warning: {ingestion_stats.get('message')}")
                    elif ingestion_stats.get("status") == "no_action":
                        st.info(f"ChromaDB ingestion: {ingestion_stats.get('message')}")
                    else: # Error
                        st.error(f"ChromaDB ingestion failed: {ingestion_stats.get('message', 'Unknown error.')}")

                    st.json(ingestion_stats) # Display full stats
                    
                    if ingestion_stats.get("chroma_final_count") is not None:
                        st.metric("Total Items in ChromaDB Collection", ingestion_stats["chroma_final_count"])
                    
                    # Optionally reset flag, or keep it to allow re-ingestion attempt if user changes settings
                    # st.session_state['successful_marker_processing'] = False 
                except Exception as e_ingest:
                    st.error(f"An critical error occurred during ChromaDB ingestion: {e_ingest}")
                    st.exception(e_ingest)

st.sidebar.markdown("---")
st.sidebar.info("Ensure `marker_single` (Marker CLI) is installed and in your system PATH. Also set `GEMINI_API_KEY` in a `.env` file or as an environment variable for the ingestion step.")