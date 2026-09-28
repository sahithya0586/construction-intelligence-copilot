import chromadb
from embeddings import generate_embeddings


# Project root
client = chromadb.PersistentClient(
    path="../../data/chroma"
)

# Use a new collection so the old 329 chunks remain safe
collection = client.get_or_create_collection(
    name="construction_documents_v2"
)


def store_embeddings():

    chunks, embeddings = generate_embeddings()

    print(f"\nStoring {len(chunks)} improved chunks in ChromaDB...")

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):

        ids.append(f"v2_chunk_{i}")

        documents.append(chunk["text"])

        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk": chunk["chunk"]
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("\n==============================")
    print("CHROMADB V2 STORAGE TEST")
    print("==============================")

    print(f"Chunks stored: {collection.count()}")


if __name__ == "__main__":
    store_embeddings()