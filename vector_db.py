# vector_db_processor.py

import os
import json
import chromadb
from tqdm import tqdm
import copy
from pathlib import Path

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("langchain library not found. Please install it: pip install langchain")

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    raise ImportError("google-generativeai library not found. Please install it: pip install google-generativeai")

from dotenv import load_dotenv

DEFAULT_CHROMA_DB_PATH_GEMINI = "./chroma_db_gemini"
DEFAULT_COLLECTION_NAME_GEMINI = "my_documents_collection_gemini"
DEFAULT_GEMINI_EMBEDDER_MODEL_ID = "embedding-001"
OUTPUT_FOLDER_PATH = "./OUTPUT"

_gemini_client_global = None
_gemini_model_id_global = None

def _initialize_gemini_client_if_needed(api_key: str, model_id: str):
    """Initializes or re-initializes the Gemini client if necessary."""
    global _gemini_client_global, _gemini_model_id_global

    if _gemini_client_global and _gemini_model_id_global == model_id:
        return True

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Cannot initialize Gemini client.")

    _gemini_model_id_global = model_id

    try:
        _gemini_client_global = genai.Client(api_key=api_key)
        test_result = _gemini_client_global.models.embed_content(
            model=model_id,
            contents=["test"],
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        if not (hasattr(test_result, 'embeddings') and test_result.embeddings and hasattr(test_result.embeddings[0], 'values')):
            if not (hasattr(test_result, 'embedding') and hasattr(test_result.embedding, 'values')):
                print("Warning: Test embedding response structure not as expected. Proceeding with caution.")
        print("Gemini Client configured successfully for embeddings.")
        return True
    except AttributeError as e:
        raise ConnectionError(f"Failed to initialize Gemini Client (AttributeError): {e}. Check google-generativeai version.")
    except Exception as e:
        raise ConnectionError(f"Failed to initialize/test Gemini Embedder with model '{model_id}': {e}")

def _sanitize_metadata(metadata_dict):
    """Sanitizes metadata to ensure compatibility with ChromaDB."""
    sanitized = {}
    if not isinstance(metadata_dict, dict):
        return {}
    for key, value in metadata_dict.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, dict)):
            try:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            except TypeError:
                sanitized[key] = str(value)
    return sanitized

langchain_chunker = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

def ingest_marker_output_to_chroma_gemini(
    marker_output_base_dir: str,
    chroma_db_path: str = DEFAULT_CHROMA_DB_PATH_GEMINI,
    collection_name: str = DEFAULT_COLLECTION_NAME_GEMINI,
    gemini_embedder_model_id: str = DEFAULT_GEMINI_EMBEDDER_MODEL_ID,
    delete_existing_collection: bool = False,
    gemini_api_key_override: str = None
):
    """Processes Marker output, embeds using Gemini, and stores in ChromaDB."""
    global _gemini_client_global

    api_key_to_use = gemini_api_key_override if gemini_api_key_override else os.getenv("GEMINI_API_KEY")
    if not api_key_to_use:
        load_dotenv()
        api_key_to_use = os.getenv("GEMINI_API_KEY")

    if not api_key_to_use:
        return {"status": "error", "message": "GEMINI_API_KEY not found."}

    try:
        _initialize_gemini_client_if_needed(api_key_to_use, gemini_embedder_model_id)
    except (ValueError, ConnectionError) as e:
        return {"status": "error", "message": str(e)}

    chroma_persistence_client = chromadb.PersistentClient(path=chroma_db_path)

    if delete_existing_collection:
        try:
            chroma_persistence_client.delete_collection(collection_name)
        except Exception:
            pass

    embedding_dim = None
    try:
        test_response = _gemini_client_global.models.embed_content(
            model=gemini_embedder_model_id,
            contents=["Determine dimension"],
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        if hasattr(test_response, 'embeddings') and test_response.embeddings and hasattr(test_response.embeddings[0], 'values'):
            embedding_dim = len(test_response.embeddings[0].values)
        else:
            test_response_single = _gemini_client_global.models.embed_content(
                model=gemini_embedder_model_id, contents="Dim single",
                config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            if hasattr(test_response_single, 'embedding') and hasattr(test_response_single.embedding, 'values'):
                embedding_dim = len(test_response_single.embedding.values)
            elif hasattr(test_response_single, 'embeddings') and test_response_single.embeddings and hasattr(test_response_single.embeddings[0], 'values'):
                embedding_dim = len(test_response_single.embeddings[0].values)

        if not embedding_dim:
            raise ValueError(f"Could not determine embedding dimension. Response: {test_response}")
    except Exception as e:
        if gemini_embedder_model_id == "embedding-001": embedding_dim = 768
        elif gemini_embedder_model_id == "text-embedding-004": embedding_dim = 768
        elif gemini_embedder_model_id == "gemini-embedding-exp-03-07": embedding_dim = 1024
        else:
            return {"status": "error", "message": f"Unknown Gemini model ('{gemini_embedder_model_id}') for fallback dimension.", "chroma_count": 0}

    collection = chroma_persistence_client.get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": f"gemini-{gemini_embedder_model_id}", "hnsw:space": "cosine"}
    )
    initial_count = collection.count()

    all_chunk_texts = []
    all_metadatas = []
    all_ids = []
    stats = {
        "processed_document_folders": 0,
        "skipped_document_folders": 0,
        "total_chunks_prepared": 0,
        "errors": [],
        "chroma_initial_count": initial_count
    }

    marker_base_path = Path(marker_output_base_dir)
    if not marker_base_path.is_dir():
        msg = f"Error: Marker output base directory not found at '{marker_base_path.resolve()}'"
        stats["errors"].append(msg)
        return {**stats, "status": "error", "message": msg, "chroma_count": initial_count}

    for item_name_path in marker_base_path.iterdir():
        if item_name_path.is_dir():
            folder_path = item_name_path
            original_doc_id = folder_path.name

            md_file_path = folder_path / f"{original_doc_id}.md"
            json_file_path = folder_path / f"{original_doc_id}_meta.json"

            if md_file_path.exists() and json_file_path.exists():
                try:
                    with open(md_file_path, 'r', encoding='utf-8') as f:
                        md_content = f.read()
                    if not md_content or md_content.isspace():
                        stats["skipped_document_folders"] += 1
                        continue

                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        raw_metadata_orig = json.load(f)
                    if not isinstance(raw_metadata_orig, dict):
                        stats["skipped_document_folders"] += 1
                        continue
                    
                    metadata_orig = _sanitize_metadata(raw_metadata_orig)
                    chunks = langchain_chunker.split_text(md_content)

                    if not chunks:
                        stats["skipped_document_folders"] += 1
                        continue

                    for i, chunk_text in enumerate(chunks):
                        chunk_id = f"{original_doc_id}_chunk_{i}"
                        chunk_metadata = copy.deepcopy(metadata_orig)
                        chunk_metadata['original_doc_id'] = original_doc_id
                        chunk_metadata['chunk_index'] = i
                        chunk_metadata['source_markdown_file'] = md_file_path.name
                        chunk_metadata['source_json_file'] = json_file_path.name

                        all_chunk_texts.append(chunk_text)
                        all_metadatas.append(chunk_metadata)
                        all_ids.append(chunk_id)
                    
                    stats["processed_document_folders"] += 1

                except json.JSONDecodeError:
                    stats["errors"].append(f"Error: Invalid JSON in file: {json_file_path.name} (folder: {original_doc_id}). Skipping.")
                    stats["skipped_document_folders"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error processing folder {original_doc_id} (path: {folder_path}): {e}. Skipping.")
                    stats["skipped_document_folders"] += 1
            else:
                stats["skipped_document_folders"] += 1

    stats["total_chunks_prepared"] = len(all_chunk_texts)

    if all_chunk_texts:
        gemini_api_batch_size = 100
        all_embeddings = []

        for i in tqdm(range(0, len(all_chunk_texts), gemini_api_batch_size), desc="Generating Gemini Embeddings"):
            batch_texts = all_chunk_texts[i : i + gemini_api_batch_size]
            try:
                response = _gemini_client_global.models.embed_content(
                    model=gemini_embedder_model_id,
                    contents=batch_texts,
                    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                if hasattr(response, 'embeddings') and response.embeddings:
                    batch_embeddings_values = [list(emb.values) for emb in response.embeddings]
                    all_embeddings.extend(batch_embeddings_values)
                else:
                    stats["errors"].append(f"Error: Unexpected response structure from embed_content for batch {i}.")
                    return {**stats, "status": "error", "message": "Unexpected response structure.", "chroma_count": collection.count()}

            except Exception as e:
                stats["errors"].append(f"Error embedding batch with Gemini (texts {i} to {i+len(batch_texts)-1}): {e}")
                return {**stats, "status": "error", "message": str(e), "chroma_count": collection.count()}
        
        if len(all_embeddings) != len(all_chunk_texts):
            stats["errors"].append(f"Error: Number of embeddings ({len(all_embeddings)}) does not match number of texts ({len(all_chunk_texts)}).")
            return {**stats, "status": "error", "message": "Embedding count mismatch.", "chroma_count": collection.count()}

        chroma_db_batch_size = 100
        for i in tqdm(range(0, len(all_ids), chroma_db_batch_size), desc="Adding chunks to ChromaDB"):
            start_idx = i
            end_idx = min(i + chroma_db_batch_size, len(all_ids))
            
            ids_batch = all_ids[start_idx:end_idx]
            embeddings_batch = all_embeddings[start_idx:end_idx]
            metadatas_batch = all_metadatas[start_idx:end_idx]
            documents_batch = all_chunk_texts[start_idx:end_idx]

            collection.add(
                ids=ids_batch,
                embeddings=embeddings_batch,
                metadatas=metadatas_batch,
                documents=documents_batch
            )
        stats["status"] = "success"
        stats["message"] = "Ingestion successful."
    elif stats["processed_document_folders"] > 0:
        stats["status"] = "warning"
        stats["message"] = "Processed document folders, but no text chunks were extracted to add to the database."
    else:
        stats["status"] = "no_action"
        stats["message"] = "No valid document folders or text chunks found to process."

    stats["chroma_final_count"] = collection.count()
    stats["chroma_items_added"] = stats["chroma_final_count"] - stats["chroma_initial_count"]
    return stats

if __name__ == "__main__":
    load_dotenv()
    api_key_standalone = os.getenv("GEMINI_API_KEY")
    if not api_key_standalone:
        print("CRITICAL: GEMINI_API_KEY not found in environment for standalone execution. Exiting.")
        exit(1)

    ingestion_results = ingest_marker_output_to_chroma_gemini(
        marker_output_base_dir=OUTPUT_FOLDER_PATH,
        chroma_db_path=DEFAULT_CHROMA_DB_PATH_GEMINI,
        collection_name=DEFAULT_COLLECTION_NAME_GEMINI,
        gemini_embedder_model_id=DEFAULT_GEMINI_EMBEDDER_MODEL_ID,
        delete_existing_collection=True,
        gemini_api_key_override=api_key_standalone
    )
    print(json.dumps(ingestion_results, indent=2))