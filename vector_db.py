import os
import json
import chromadb
import logging
from tqdm import tqdm
import copy
import json 
import numpy as np # Sentence Transformers often returns numpy arrays

# --- Use standard libraries ---
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
# Make sure this points to the parent directory of your subfolders
OUTPUT_FOLDER_PATH = "./OUTPUT"  # <<< Adjusted based on image (assuming it's relative path)
CHROMA_DB_PATH = "./chroma_db_standard"
COLLECTION_NAME = "my_documents_collection_standard"

def sanitize_metadata(metadata_dict):
    """
    Recursively sanitizes a dictionary to ensure all values are ChromaDB-compatible
    (str, int, float, bool). Converts lists/dicts to JSON strings.
    """
    sanitized = {}
    if not isinstance(metadata_dict, dict):
        # If the top level isn't a dict, return empty or handle appropriately
        return {}
    for key, value in metadata_dict.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, dict)):
            # Convert lists and nested dictionaries to JSON strings
            try:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            except TypeError:
                # Fallback if JSON serialization fails (e.g., non-serializable objects)
                sanitized[key] = str(value)
    return sanitized

# --- Chunking Configuration (using Langchain) ---
try:
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    print(f"Using Langchain RecursiveCharacterTextSplitter: size={chunker._chunk_size}, overlap={chunker._chunk_overlap}")
except ImportError:
    print("langchain library not found. Install it: pip install langchain")
    exit(1)
except Exception as e:
    print(f"Error initializing Langchain text splitter: {e}")
    exit(1)


# --- Embedder Configuration (using Sentence Transformers) ---
try:
    EMBEDDER_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device="cuda") # Use 'cuda' if available
    print(f"Using Sentence Transformer Embedder: {EMBEDDER_MODEL_NAME}")
except ImportError:
    print("sentence-transformers library not found. Install it: pip install sentence-transformers")
    exit(1)
except Exception as e:
    print(f"Error initializing Sentence Transformer: {e}")
    exit(1)


# --- Initialize ChromaDB ---
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Attempted to delete existing collection (if any): {COLLECTION_NAME}")
except Exception:
    # Collection might not exist, which is fine
    pass

try:
    embedding_dim = embedder.get_sentence_embedding_dimension()
    print(f"Detected embedding dimension: {embedding_dim}")
except Exception:
    try:
        dummy_embedding = embedder.encode("test")
        embedding_dim = len(dummy_embedding)
        print(f"Detected embedding dimension via dummy text: {embedding_dim}")
    except Exception as e2:
        print(f"Could not determine embedding dimension: {e2}")
        exit(1)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"embedding_model": EMBEDDER_MODEL_NAME}
)
print(f"ChromaDB collection '{COLLECTION_NAME}' ready.")


# --- Data Preparation (with Chunking using Langchain) ---
all_chunk_texts = []
all_metadatas = []
all_ids = []

print(f"\nScanning directory: {os.path.abspath(OUTPUT_FOLDER_PATH)}") # Show absolute path

subfolders_processed = 0
subfolders_skipped = 0

# Ensure the OUTPUT folder exists
if not os.path.isdir(OUTPUT_FOLDER_PATH):
    print(f"Error: Output directory not found at '{os.path.abspath(OUTPUT_FOLDER_PATH)}'")
    exit(1)

# Iterate through items in the output folder
for item_name in os.listdir(OUTPUT_FOLDER_PATH):
    item_path = os.path.join(OUTPUT_FOLDER_PATH, item_name)

    if os.path.isdir(item_path):
        # print(f"Processing directory: {item_name}") # Uncomment for debug
        folder_path = item_path
        md_file_path = None
        json_file_path = None
        original_doc_id = item_name # Use folder name as base ID

        try:
            found_md = False
            found_json = False
            # Look for files directly inside the subfolder
            for file_name in os.listdir(folder_path):
                file_path_full = os.path.join(folder_path, file_name)
                if not os.path.isfile(file_path_full): # Skip sub-sub-folders etc.
                    continue

                # Check for Markdown file (ending in .md or .markdown)
                if (file_name.lower().endswith(".md") or file_name.lower().endswith(".markdown")) and not found_md:
                     # print(f"  Found MD file: {file_name}") # Uncomment for debug
                     md_file_path = file_path_full
                     found_md = True

                elif file_name.lower().endswith("_meta.json") and not found_json:
                    # print(f"  Found JSON file: {file_name}") # Uncomment for debug
                    json_file_path = file_path_full
                    found_json = True
                
                # Optimization: stop looking once both are found
                if found_md and found_json:
                    break # Stop searching this folder once both are found

        except FileNotFoundError:
            print(f"Warning: Folder '{item_name}' became inaccessible? Skipping.")
            subfolders_skipped += 1
            continue
        except Exception as e:
            print(f"Error listing files in folder {item_name} ({folder_path}): {e}. Skipping.")
            subfolders_skipped += 1
            continue

        # Proceed only if BOTH files were found by the modified logic
        if md_file_path and json_file_path:
            try:
                
                with open(md_file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()

                if not md_content or md_content.isspace():
                    print(f"Skipping folder {item_name}: Markdown file '{os.path.basename(md_file_path)}' is empty.")
                    subfolders_skipped += 1
                    continue

                # Load the raw metadata
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    raw_metadata_orig = json.load(f) # Load raw data first

                # Validate if it's a dictionary BEFORE sanitizing
                if not isinstance(raw_metadata_orig, dict):
                     print(f"Warning: Metadata in {os.path.basename(json_file_path)} (folder: {item_name}) is not a dictionary. Skipping.")
                     subfolders_skipped += 1
                     continue

                metadata_orig = sanitize_metadata(raw_metadata_orig)
                

                chunks = chunker.split_text(md_content)

                if not chunks:
                     print(f"Warning: Chunker returned no chunks for {os.path.basename(md_file_path)} (folder: {item_name}). Skipping.")
                     subfolders_skipped += 1
                     continue

                # Process each chunk
                for i, chunk_text in enumerate(chunks):
                    chunk_id = f"{original_doc_id}_chunk_{i}"

                    # Create metadata for the chunk: Copy SANITIZED original, add chunk info
                    chunk_metadata = copy.deepcopy(metadata_orig) # Now copying the sanitized dict
                    chunk_metadata['original_doc_id'] = original_doc_id
                    chunk_metadata['chunk_index'] = i
                    chunk_metadata['source_markdown_file'] = os.path.basename(md_file_path)
                    chunk_metadata['source_json_file'] = os.path.basename(json_file_path)

                    all_chunk_texts.append(chunk_text)
                    all_metadatas.append(chunk_metadata) # Append the sanitized+augmented dict
                    all_ids.append(chunk_id)

                subfolders_processed += 1

            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in file: {os.path.basename(json_file_path)} (folder: {item_name}). Skipping.")
                subfolders_skipped += 1
            except Exception as e:
                print(f"Error processing folder {item_name} (path: {folder_path}): {e}. Skipping.")
                subfolders_skipped += 1
        else:
            subfolders_skipped += 1
    


print(f"\nScan complete.")
print(f"--> Processed {subfolders_processed} folders successfully.")
print(f"--> Skipped {subfolders_skipped} items (non-directories or folders missing required files/content/valid JSON).")
print(f"--> Total chunks prepared: {len(all_chunk_texts)}")


if all_chunk_texts:
    print(f"\nEmbedding {len(all_chunk_texts)} text chunks using {EMBEDDER_MODEL_NAME}...")
    try:
        embeddings_np = embedder.encode(all_chunk_texts, show_progress_bar=True)
        embeddings = embeddings_np.tolist()
        print("Embeddings generated successfully.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        exit(1)

    print(f"\nAdding {len(all_ids)} chunks to ChromaDB collection '{COLLECTION_NAME}'...")
    try:
        batch_size = 100
        for i in tqdm(range(0, len(all_ids), batch_size), desc="Adding chunks to ChromaDB"):
             start_idx = i
             end_idx = min(i + batch_size, len(all_ids))
             ids_batch = all_ids[start_idx:end_idx]
             embeddings_batch = embeddings[start_idx:end_idx]
             metadatas_batch = all_metadatas[start_idx:end_idx]
             documents_batch = all_chunk_texts[start_idx:end_idx]

             collection.add(
                 ids=ids_batch,
                 embeddings=embeddings_batch,
                 metadatas=metadatas_batch,
                 documents=documents_batch
             )
        print("\nData added to ChromaDB successfully!")
    except Exception as e:
        print(f"Error adding data to ChromaDB: {e}")

else:
    print("\nNo valid text chunks found to add to the database.")
