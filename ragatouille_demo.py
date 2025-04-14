import requests
import time
from ragatouille import RAGPretrainedModel

def get_wikipedia_page(title: str) -> str:
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: Title of the Wikipedia page.
    :return: Extracted text content as a string.
    """
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    headers = {
        "User-Agent": "RAGatouille_tutorial/0.0.1 (contact@example.com)"
    }
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def main():
    print("🔄 Loading RAGPretrainedModel...")
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    print("📄 Fetching Hayao Miyazaki Wikipedia page...")
    full_document = get_wikipedia_page("Hayao_Miyazaki")
    print(f"📏 Document length: {len(full_document)} characters")

    print("📦 Indexing document...")
    RAG.index(
        collection=[full_document],
        document_ids=["miyazaki"],
        document_metadatas=[{"entity": "person", "source": "wikipedia"}],
        index_name="Miyazaki",
        max_document_length=180,
        split_documents=True,
    )
    print("✅ Indexing complete!\n")

    query = "What animation studio did Miyazaki found?"
    print(f"🔍 Querying: \"{query}\"")
    results = RAG.search(query=query, k=3)

    for result in results:
        print("\n---")
        print(f"Rank: {result['rank']}")
        print(f"Score: {result['score']:.2f}")
        print(f"Document ID: {result['document_id']}")
        print(f"Metadata: {result['document_metadata']}")
        print(f"Content:\n{result['content'][:500]}...")

    print("\n⏱ Timing single search...")
    start = time.time()
    RAG.search(query=query)
    duration_ms = (time.time() - start) * 1000
    print(f"Search took {duration_ms:.2f} ms")

    print("\n🔁 Running batch queries...")
    batch_queries = [
        "What animation studio did Miyazaki found?",
        "Miyazaki son name"
    ]
    batch_results = RAG.search(query=batch_queries, k=2)
    for i, result_list in enumerate(batch_results):
        print(f"\n📌 Results for: \"{batch_queries[i]}\"")
        for res in result_list:
            print(f"  Rank {res['rank']} - Score: {res['score']:.2f}")
            print(f"  {res['content'][:300]}...\n")


if __name__ == "__main__":
    main()
