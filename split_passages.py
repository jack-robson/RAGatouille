import requests
from ragatouille.data.corpus_processor import CorpusProcessor

def get_wikipedia_page(title: str) -> str:
    """
    Retrieve the full text content of a Wikipedia page.
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
    print("📄 Fetching Hayao Miyazaki Wikipedia page...")
    full_document = get_wikipedia_page("Hayao_Miyazaki")
    print(f"📏 Document length: {len(full_document)} characters")

    print("🪓 Splitting document into passages...")
    processor = CorpusProcessor()
    passages = processor.process_corpus(
        documents=[full_document],
        document_ids=["miyazaki"],
        chunk_size=180
    )

    for i, p in enumerate(passages):
        print(f"\n--- Passage {i + 1} ---")
        print(p["content"][:500] + ("..." if len(p["content"]) > 500 else ""))


if __name__ == "__main__":
    main()
