import chromadb
from sentence_transformers import SentenceTransformer


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="../../data/chroma"
)

# Use improved V2 collection
collection = client.get_collection(
    name="construction_documents_v2"
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def search_documents(query, n_results=5):

    # Convert question into embedding
    query_embedding = model.encode([query])[0]

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    return results


if __name__ == "__main__":

    query = input("\nAsk a construction question: ")

    results = search_documents(query)

    print("\n==============================")
    print("IMPROVED SEMANTIC SEARCH")
    print("==============================")

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        print(f"\n--- Result {i + 1} ---")

        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Chunk: {metadata['chunk']}")

        print("\nText:")
        print(document[:1000])