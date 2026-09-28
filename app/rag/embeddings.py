from chunker import create_chunks
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    chunks = create_chunks()

    texts = [chunk["text"] for chunk in chunks]

    print(f"\nGenerating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    print("\n==============================")
    print("EMBEDDING TEST")
    print("==============================")

    print(f"Number of embeddings: {len(embeddings)}")

    if len(embeddings) > 0:
        print(f"Embedding dimensions: {len(embeddings[0])}")
        print("\nFirst embedding:")
        print(embeddings[0][:10])

    return chunks, embeddings


if __name__ == "__main__":
    generate_embeddings()