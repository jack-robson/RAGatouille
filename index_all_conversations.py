import json
import time
from pathlib import Path
from glob import glob
from collections import defaultdict
from ragatouille import RAGPretrainedModel
from tqdm import tqdm

DATA_DIR = Path("./data")
INDEX_NAME = "Me"
INDEX_DIR = Path(".ragatouille/colbert/indexes") / INDEX_NAME

chat_title_counts = defaultdict(int)

# ------- Document ID Function -------- #
def generate_unique_document_ids(chunks):
    doc_ids = []

    for chunk in chunks:
        title = chunk["chatTitle"].replace(" ", "_").lower()
        chat_title_counts[title] += 1
        session_id = chat_title_counts[title]
        doc_id = f"{title}_{session_id}_chunk_{chunk['chunkNo']}"
        doc_ids.append(doc_id)

    return doc_ids


# ------- Load a single .chunked.json -------- #
def load_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = [chunk["chunkText"] for chunk in data]
    document_ids = generate_unique_document_ids(data)
    metadatas = [
        {
            "chatTitle": chunk["chatTitle"],
            "chunkNo": chunk["chunkNo"],
            "tokenCount": chunk["tokenCount"]
        }
        for chunk in data
    ]
    return documents, document_ids, metadatas


# ------- Load all files into one corpus -------- #
def load_all_chunks():
    print("🔍 Scanning for chunked data files...")
    paths = sorted(glob(str(DATA_DIR / "conversation_*.chunked.json")))
    print(f"📂 Found {len(paths)} files.")

    all_docs = []
    all_ids = []
    all_metas = []

    start = time.time()
    for i, path in enumerate(tqdm(paths, desc="📦 Loading & preparing chunks")):
        docs, ids, metas = load_chunks(path)
        all_docs.extend(docs)
        all_ids.extend(ids)
        all_metas.extend(metas)

        # Estimate time remaining
        elapsed = time.time() - start
        per_file = elapsed / (i + 1)
        eta = per_file * (len(paths) - (i + 1))
        print(f"⏳ ETA: {eta:.1f}s remaining", end="\r")

    return all_docs, all_ids, all_metas


# ------- Main Indexing Logic -------- #
def main():
    if INDEX_DIR.exists():
        print(f"✅ Index '{INDEX_NAME}' already exists. Skipping indexing.")
        return

    print("🔄 Loading RAGPretrainedModel...")
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    documents, document_ids, metadatas = load_all_chunks()

    print(f"\n📊 Total documents: {len(documents)}")
    print("🧠 Starting indexing...\n")

    RAG.index(
        collection=documents,
        document_ids=document_ids,
        document_metadatas=metadatas,
        index_name=INDEX_NAME,
        max_document_length=180,
        split_documents=False
    )

    print("✅ Indexing complete! Index name:", INDEX_NAME)


if __name__ == "__main__":
    main()
