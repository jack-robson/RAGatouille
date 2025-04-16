import json
from pathlib import Path
from ragatouille import RAGPretrainedModel

def main():
    print("Welcome to the semantic signal mining tool!")
    print("This script will guide you through providing queries, options, and saving results.\n")

    # 1. Ask for index name and location:
    index_name = input("Enter the name of the index (default: Me): ").strip() or "Me"
    index_dir_input = input("Enter the path to the directory containing the index (default: .ragatouille/colbert/indexes): ").strip()
    index_dir = Path(index_dir_input) if index_dir_input else Path(".ragatouille/colbert/indexes")
    index_path = index_dir / index_name

    if not index_path.exists():
        print(f"❌ Index '{index_name}' not found at: {index_path}")
        print("Exiting script.")
        return

    # 2. Prompt user for number of queries and gather them
    while True:
        try:
            num_queries = int(input("\nHow many queries do you want to provide? (e.g., 1, 2, 3...): "))
            if num_queries < 1:
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive integer.")

    queries = []
    for i in range(num_queries):
        q = input(f"Enter query #{i+1}: ").strip()
        queries.append(q)

    # 3. Ask for top_k
    while True:
        top_k_input = input("\nEnter the number of search results to retrieve for each query (top_k) [default: 20]: ").strip()
        if not top_k_input:
            top_k = 20
            break
        try:
            top_k = int(top_k_input)
            if top_k < 1:
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive integer.")

    # 4. Ask for score threshold
    while True:
        threshold_input = input("Enter the minimum score threshold for displaying results [default: 15]: ").strip()
        if not threshold_input:
            score_threshold = 15
            break
        try:
            score_threshold = float(threshold_input)
            break
        except ValueError:
            print("Please enter a valid number.")

    # 5. Load the RAG model
    print(f"\n✅ Using existing index: '{index_name}' at {index_path}")
    RAG = RAGPretrainedModel.from_index(index_path)

    # 6. Perform the searches
    print(f"\n🔍 Running semantic signal mining (score threshold = {score_threshold}, top_k = {top_k})...\n")

    all_queries_results = []

    for query in queries:
        print(f"🧠 Query: \"{query}\"")
        results = RAG.search(query=query, k=top_k)
        
        # Filter results
        filtered_results = [r for r in results if r["score"] > score_threshold]

        # Store in a data structure for possible output
        current_query_data = {
            "query": query,
            "results": filtered_results
        }
        all_queries_results.append(current_query_data)

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
            content_excerpt = result['content'][:500]
            print(f"Content:\n{content_excerpt}...")

        print("\n" + "=" * 60 + "\n")

    # 7. Optionally save to file
    save_choice = input("Do you want to save the results to a JSON file? (y/N): ").strip().lower()
    if save_choice == "y":
        output_file = input("Enter the path/filename to save results (e.g., results.json): ").strip()
        if not output_file:
            output_file = "results.json"
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_queries_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to '{output_path.absolute()}'")
    else:
        print("Results were not saved to a file.")

    print("\nDone! Thank you for using the semantic signal mining tool.")

if __name__ == "__main__":
    main()
