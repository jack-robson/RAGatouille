import json
from pathlib import Path
from ragatouille import RAGPretrainedModel

INDEX_NAME = "Me"
INDEX_DIR = Path(".ragatouille/colbert/indexes") / INDEX_NAME
TOP_K = 20
SCORE_THRESHOLD = 15

# High-signal, semantically rich mining prompts
SIGNAL_QUERIES = [
    "Messages where the speaker expresses strong frustration",
    "Parts of the chat where the speaker wishes something existed",
    "Examples of someone obsessing over a problem or idea",
    "Complaints about inefficient tools or workflows",
    "Venting about things that take too much time",
    "Manual workarounds that feel annoying or broken",
    "Recurring problems that keep coming up",
    "Moments when the speaker is clearly annoyed",
    "Someone expressing emotional stress or dissatisfaction",
    "Discussions about needs that aren't being met"
]

def main():
    if not INDEX_DIR.exists():
        print(f"❌ Index '{INDEX_NAME}' not found at: {INDEX_DIR}")
        return

    print(f"✅ Using existing index: '{INDEX_NAME}'")
    RAG = RAGPretrainedModel.from_index(INDEX_DIR)

    print(f"\n🔍 Running semantic signal mining queries (only showing results with score > {SCORE_THRESHOLD}):\n")

    for query in SIGNAL_QUERIES:
        print(f"🧠 Query: \"{query}\"")
        results = RAG.search(query=query, k=TOP_K)

        filtered_results = [r for r in results if r["score"] > SCORE_THRESHOLD]

        if not filtered_results:
            print("😕 No strong matches above threshold.\n")
            print("=" * 60 + "\n")
            continue

        for result in filtered_results:
            print("\n---")
            print(f"Rank: {result['rank']}")
            print(f"Score: {result['score']:.2f}")
            print(f"Document ID: {result['document_id']}")
            print(f"Metadata: {result['document_metadata']}")
            print(f"Content:\n{result['content'][:500]}...")

        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()
