import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Image metadata directory
IMAGE_METADATA_DIR = PROJECT_ROOT / "data" / "image_metadata"

# ChromaDB
client = chromadb.PersistentClient(
    path=str(PROJECT_ROOT / "data" / "chroma")
)

collection = client.get_collection(
    name="construction_documents_v2"
)

# Same embedding model used by the PDF pipeline
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_image_text(analysis):
    """Convert structured VLM output into searchable text."""

    parts = [
        f"Image summary: {analysis.get('image_summary', '')}",
        f"Construction phase: {analysis.get('construction_phase', '')}",
        f"Activities: {', '.join(analysis.get('activities', []))}",
        f"Materials: {', '.join(analysis.get('materials', []))}",
        f"Equipment: {', '.join(analysis.get('equipment', []))}",
        f"Structural elements: {', '.join(analysis.get('structural_elements', []))}",
        f"Workers: {', '.join(analysis.get('workers', []))}",
        f"Safety hazards: {', '.join(analysis.get('safety_hazards', []))}",
        f"Site conditions: {', '.join(analysis.get('site_conditions', []))}",
        f"Progress indicators: {', '.join(analysis.get('progress_indicators', []))}",
    ]

    return "\n".join(parts)


def add_image_embeddings():

    json_files = sorted(IMAGE_METADATA_DIR.glob("*.json"))

    print(f"Image metadata files found: {len(json_files)}")

    if not json_files:
        print("No image metadata files found.")
        return

    documents = []
    ids = []
    metadatas = []

    for json_file in json_files:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_id = data["image_id"]
        analysis = data["analysis"]

        document = build_image_text(analysis)

        ids.append(f"image_{image_id}")
        documents.append(document)

        metadatas.append({
            "source_type": "image",
            "image_id": image_id,
            "image_filename": data["image_filename"],
            "source": data["image_path"],
            "construction_phase": analysis.get(
                "construction_phase", ""
            )
        })

    print(f"Generating embeddings for {len(documents)} images...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    )

    print("Adding image embeddings to ChromaDB...")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("\n==============================")
    print("IMAGE EMBEDDINGS COMPLETE")
    print("==============================")

    print(f"ChromaDB records: {collection.count()}")
    print(f"Images added/updated: {len(ids)}")


if __name__ == "__main__":
    add_image_embeddings()
