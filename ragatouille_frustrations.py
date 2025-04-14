import json
import time
from pathlib import Path
from collections import defaultdict
from ragatouille import RAGPretrainedModel

DATA_PATH = Path("data/conversation_001.chunked.json")
INDEX_NAME = "PersonalConversation001"
QUERY = "frustrations"
TOP_K = 10


def generate_unique_document_ids(chunks):
    chat_title_counts = defaultdict(int)
    doc_ids = []

    for chunk in chunks:
        title = chunk["chatTitle"].replace(" ", "_").lower()
        chat_title_counts[title] += 1
        session_id = chat_title_counts[title]
        doc_id = f"{title}_{session_id}_chunk_{chunk['chunkNo']}"
        doc_ids.append(doc_id)

    return doc_ids


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


def main():
    print("🔄 Loading RAGPretrainedModel...")
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    print(f"📄 Loading personal data from {DATA_PATH}...")
    documents, document_ids, metadatas = load_chunks(DATA_PATH)
    print(f"📦 Indexing {len(documents)} chunks...")

    RAG.index(
        collection=documents,
        document_ids=document_ids,
        document_metadatas=metadatas,
        index_name=INDEX_NAME,
        max_document_length=180,
        split_documents=False  # Already pre-chunked!
    )

    print("✅ Indexing complete!\n")

    print(f"🔍 Searching for: \"{QUERY}\" (top {TOP_K} results)")
    results = RAG.search(query=QUERY, k=TOP_K)

    for result in results:
        print("\n---")
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Document ID: {result['document_id']}")
        print(f"Metadata: {result['document_metadata']}")
        print(f"Content:\n{result['content'][:500]}...")


if __name__ == "__main__":
    main()
