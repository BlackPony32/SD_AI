import asyncio
import hashlib
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import tiktoken
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

# --- Configuration ---
DB_PATH = "./chroma_data"
COLLECTION_NAME = "md_collection"
CHROMA_SETTINGS = Settings(allow_reset=True)
MAX_TOKENS = 500
ENCODING = tiktoken.get_encoding("cl100k_base")

# --- Utility functions ---
def num_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

def split_long_text(text: str, max_tokens: int, base_metadata: dict):
    """
    Split text into chunks not exceeding max_tokens.
    base_metadata should contain 'block_id' and any fixed fields.
    Returns list of (chunk_text, metadata) where metadata includes 'chunk_num'.
    """
    if num_tokens(text) <= max_tokens:
        return [(text, {**base_metadata, "chunk_num": 1, "total_chunks": 1})]
    
    # Simple sentence splitter (improve as needed)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent_len = num_tokens(sent)
        if current_len + sent_len <= max_tokens:
            current_chunk.append(sent)
            current_len += sent_len
        else:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
            current_chunk = [sent]
            current_len = sent_len
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)
    
    # Add metadata for each chunk
    total = len(chunks)
    result = []
    for idx, chunk_text in enumerate(chunks, 1):
        meta = {**base_metadata, "chunk_num": idx, "total_chunks": total}
        result.append((chunk_text, meta))
    return result

def parse_faq_only(file_path: str):
    """
    Parse the FAQ file and return a list of chunks with metadata.
    Does not interact with ChromaDB.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_chunks = []  # will hold (text, metadata)
    current_category = "General"
    current_block_lines = []
    in_question = False
    q_pattern = re.compile(r'^Q[.:?]|^\*\*Q[.:?]')

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue

        if line.startswith("### "):
            # Save previous block
            if current_block_lines:
                block_text = "\n".join(current_block_lines).strip()
                if q_pattern.match(block_text):
                    block_id = hashlib.md5(block_text.encode()).hexdigest()
                    base_meta = {"source": file_path, "category": current_category, "block_id": block_id}
                    chunks = split_long_text(block_text, MAX_TOKENS, base_meta)
                    all_chunks.extend(chunks)
                current_block_lines = []
                in_question = False
            current_category = line.replace("### ", "").replace("**", "").strip()

        elif q_pattern.match(line):
            if current_block_lines:
                block_text = "\n".join(current_block_lines).strip()
                if q_pattern.match(block_text):
                    block_id = hashlib.md5(block_text.encode()).hexdigest()
                    base_meta = {"source": file_path, "category": current_category, "block_id": block_id}
                    chunks = split_long_text(block_text, MAX_TOKENS, base_meta)
                    all_chunks.extend(chunks)
            current_block_lines = [line]
            in_question = True

        elif in_question:
            current_block_lines.append(line)

        # else ignore

    # Last block
    if current_block_lines:
        block_text = "\n".join(current_block_lines).strip()
        if q_pattern.match(block_text):
            block_id = hashlib.md5(block_text.encode()).hexdigest()
            base_meta = {"source": file_path, "category": current_category, "block_id": block_id}
            chunks = split_long_text(block_text, MAX_TOKENS, base_meta)
            all_chunks.extend(chunks)

    return all_chunks

def analyze_chunks(chunks):
    """
    Given list of (text, metadata), compute statistics and plot distributions.
    Also print examples of split blocks.
    """
    texts = [t for t, _ in chunks]
    metas = [m for _, m in chunks]
    
    token_counts = [num_tokens(t) for t in texts]
    char_counts = [len(t) for t in texts]
    
    print("\nCHUNK STATISTICS")
    print(f"Total chunks: {len(chunks)}")
    print(f"Token counts: min={min(token_counts)}, max={max(token_counts)}, mean={np.mean(token_counts):.1f}, median={np.median(token_counts)}")
    print(f"Char counts : min={min(char_counts)}, max={max(char_counts)}, mean={np.mean(char_counts):.1f}, median={np.median(char_counts)}")
    
    # Identify blocks that were split (total_chunks > 1)
    split_blocks = defaultdict(list)
    for text, meta in chunks:
        if meta.get("total_chunks", 1) > 1:
            split_blocks[meta["block_id"]].append((meta["chunk_num"], text))
    
    print(f"\nBlocks that were split: {len(split_blocks)}")
    # Show a few examples
    for bid, parts in list(split_blocks.items())[:3]:  # show up to 3
        parts.sort()
        print(f"\n--- Block {bid[:8]}... split into {len(parts)} chunks ---")
        for i, (_, text) in enumerate(parts, 1):
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"Chunk {i}: {preview}")
    
    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(token_counts, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_title("Token Count Distribution")
    axes[0].set_xlabel("Tokens")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(MAX_TOKENS, color='red', linestyle='--', label=f"Max tokens ({MAX_TOKENS})")
    axes[0].legend()
    
    axes[1].hist(char_counts, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_title("Character Count Distribution")
    axes[1].set_xlabel("Characters")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def format_search_results(results: list) -> str:
    """Formats a list of database results into a single readable string for the LLM context."""
    if not results:
        return "No results found."

    output_lines = []
    
    for i, r in enumerate(results, 1):
        output_lines.append(f"{'='*60}")
        
        # Safely extract metadata
        metadata = r.get('metadata', {})
        
        # Handle distance
        distance = r.get('distance')
        try:
            dist_str = f"{float(distance):.4f}" if distance is not None else "N/A"
        except (ValueError, TypeError):
            dist_str = str(distance)
            
        output_lines.append(f"Result {i} | Distance: {dist_str}")
        output_lines.append(f"Category: {metadata.get('category', 'N/A')}")
        
        # Add chunk info if it exists
        if metadata.get('total_chunks', 1) > 1:
            chunk_num = metadata.get('chunk_num', '?')
            total_chunks = metadata.get('total_chunks', '?')
            output_lines.append(f"Part {chunk_num} of {total_chunks}")
            
        output_lines.append(f"{'-'*60}")
        
        # Add the FULL chunk text, ensuring it is a string
        text = str(r.get('text', ''))
        output_lines.append(text)
        
        # Clean blank line between results
        output_lines.append("") 
        
    return "\n".join(output_lines).strip()

# --- ChromaDB functions ---
async def clean_db(db_path: str = DB_PATH):
    def _reset():
        client = chromadb.PersistentClient(path=db_path, settings=CHROMA_SETTINGS)
        client.reset()
        print(f"Database at '{db_path}' has been completely reset.")
    await asyncio.to_thread(_reset)

def init_and_load_md(file_path: str, db_path: str = DB_PATH, collection_name: str = COLLECTION_NAME):
    """Parse, split, and load into ChromaDB (with block_id)."""
    def _check_and_load():
        client = chromadb.PersistentClient(path=db_path, settings=CHROMA_SETTINGS)

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )

        collection = client.get_or_create_collection(
            name=collection_name, 
            embedding_function=openai_ef
        )
        
        if collection.count() > 0:
            print(f"Collection '{collection_name}' already has {collection.count()} chunks. Skipping parsing.")
            return True

        # Parse using the same logic (but without loading into DB)
        chunks_with_meta = parse_faq_only(file_path)
        if not chunks_with_meta:
            print("No valid Q&A blocks found.")
            return False
        
        documents = [text for text, _ in chunks_with_meta]
        metadatas = [meta for _, meta in chunks_with_meta]
        ids = [hashlib.md5(text.encode()).hexdigest() for text in documents]
        
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Successfully loaded {len(documents)} chunks into '{collection_name}'.")
        return True

    _check_and_load()

def search_md_db(query_text: str, n_results: int = 5, db_path: str = DB_PATH, collection_name: str = COLLECTION_NAME):
    def _search():
        client = chromadb.PersistentClient(path=db_path, settings=CHROMA_SETTINGS)
        collection = client.get_or_create_collection(name=collection_name)
        if collection.count() == 0:
            return []
        results = collection.query(query_texts=[query_text], n_results=n_results)
        formatted = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        return formatted
    return _search()

