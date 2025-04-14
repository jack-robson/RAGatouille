import json
import time
from pathlib import Path
from collections import defaultdict
from ragatouille import RAGPretrainedModel

DATA_PATH = Path("data/conversation_001.chunked.json")
INDEX_NAME = "PersonalConversation001"
INDEX_DIR = Path(".ragatouille/colbert/indexes") / INDEX_NAME
TOP_K = 10

# Frustration/obsession/rant/idea mining prompts
MINING_QUERIES = [
    "What does he keep complaining about?",
    "What recurring problems or rants show up across conversations?",
    "What ideas or thoughts keep coming up?",
    "What’s something he seems obsessed with?",
    "What causes emotional frustration or stress?",
    "What are the repeated emotional themes?",
    "Which parts of the conversation feel intense or dramatic?",
    "What issues keep getting revisited?",
    "Where does he vent or express dissatisfaction?",
    "Which goals or problems come up over and over?"
]


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
    if INDEX_DIR.exists():
        print(f"✅ Found existing index: '{INDEX_NAME}'")
        RAG = RAGPretrainedModel.from_index(INDEX_DIR)  # ✅ full path
    else:
        print("🔄 Loading RAGPretrainedModel and preparing new index...")
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
            split_documents=False
        )
        print("✅ Indexing complete!\n")

    print(f"\n🔍 Running mining queries (top {TOP_K} each):\n")

    for query in MINING_QUERIES:
        print(f"🧠 Query: \"{query}\"")
        results = RAG.search(query=query, k=TOP_K)

        if not results:
            print("⚠️ No results returned.")
            continue

        for result in results:
            print("\n---")
            print(f"Rank: {result['rank']}")
            print(f"Score: {result['score']:.2f}")
            print(f"Document ID: {result['document_id']}")
            print(f"Metadata: {result['document_metadata']}")
            print(f"Content:\n{result['content'][:500]}...")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
