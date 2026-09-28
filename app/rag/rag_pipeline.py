import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# 1. PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 2. GEMINI SETUP
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )

gemini_client = genai.Client(
    api_key=api_key
)

GEMINI_MODEL = "gemini-3.8-flash"


# ============================================================
# 3. CHROMADB SETUP
# ============================================================

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_collection(
    name="construction_documents_v2"
)


# ============================================================
# 4. EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 5. QUESTION TYPE DETECTION
# ============================================================

def detect_question_type(query):

    query_lower = query.lower()

    image_keywords = [
        "image",
        "photo",
        "picture",
        "visible",
        "shown",
        "shown in",
        "see in",
        "site image",
        "site photo",
        "construction site image",
        "construction site photo",
        "scaffolding",
        "equipment visible",
        "hazards visible",
    ]

    pdf_keywords = [
        "pdf",
        "document",
        "documents",
        "guideline",
        "guidelines",
        "recommendation",
        "recommendations",
        "according to",
        "standard",
        "procedure",
        "procedures",
        "safety program",
        "policy",
        "what does the document",
    ]

    has_image = any(
        keyword in query_lower
        for keyword in image_keywords
    )

    has_pdf = any(
        keyword in query_lower
        for keyword in pdf_keywords
    )

    if has_image and has_pdf:
        return "mixed"

    if has_image:
        return "image"

    if has_pdf:
        return "pdf"

    return "general"


# ============================================================
# 6. SOURCE TYPE
# ============================================================

def get_source_type(metadata):

    if metadata.get("source_type") == "image":
        return "image"

    return "pdf"


# ============================================================
# 7. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query, n_results=8):

    question_type = detect_question_type(query)

    MAX_DISTANCE = 0.85


    # --------------------------------------------------------
    # SEARCH FUNCTION
    # --------------------------------------------------------

    def run_search(search_query, source_type=None):

        query_embedding = embedding_model.encode(
            [search_query]
        )[0]

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=n_results
        )

        selected_documents = []
        selected_metadatas = []
        selected_distances = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            if distance > MAX_DISTANCE:
                continue

            current_source_type = get_source_type(
                metadata
            )

            if source_type is not None:

                if current_source_type != source_type:
                    continue

            selected_documents.append(
                document
            )

            selected_metadatas.append(
                metadata
            )

            selected_distances.append(
                distance
            )

        return (
            selected_documents,
            selected_metadatas,
            selected_distances
        )


    # ========================================================
    # IMAGE QUESTION
    # ========================================================

    if question_type == "image":

        documents, metadatas, distances = run_search(
            query,
            source_type="image"
        )


    # ========================================================
    # PDF QUESTION
    # ========================================================

    elif question_type == "pdf":

        documents, metadatas, distances = run_search(
            query,
            source_type="pdf"
        )


    # ========================================================
    # MIXED QUESTION
    # ========================================================

    elif question_type == "mixed":

        image_query = (
            "construction site image visible safety hazards "
            "workers scaffolding concrete building rebar "
            "construction activities site conditions"
        )

        pdf_query = (
            "construction safety hazard identification "
            "assessment prevention inspections hazard control "
            "safety procedures worker protection"
        )

        image_documents, image_metadatas, image_distances = run_search(
            image_query,
            source_type="image"
        )

        pdf_documents, pdf_metadatas, pdf_distances = run_search(
            pdf_query,
            source_type="pdf"
        )

        # Keep strongest results from each source
        image_documents = image_documents[:3]
        image_metadatas = image_metadatas[:3]
        image_distances = image_distances[:3]

        pdf_documents = pdf_documents[:3]
        pdf_metadatas = pdf_metadatas[:3]
        pdf_distances = pdf_distances[:3]

        documents = (
            image_documents +
            pdf_documents
        )

        metadatas = (
            image_metadatas +
            pdf_metadatas
        )

        distances = (
            image_distances +
            pdf_distances
        )


    # ========================================================
    # GENERAL QUESTION
    # ========================================================

    else:

        documents, metadatas, distances = run_search(
            query
        )


    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
        "question_type": question_type
    }


# ============================================================
# 8. BUILD CONTEXT
# ============================================================

def build_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []

    image_number = 0
    pdf_number = 0


    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        source_type = get_source_type(
            metadata
        )


        # ====================================================
        # IMAGE
        # ====================================================

        if source_type == "image":

            image_number += 1

            image_filename = metadata.get(
                "image_filename",
                "Unknown image"
            )

            image_id = metadata.get(
                "image_id",
                "Unknown image ID"
            )

            construction_phase = metadata.get(
                "construction_phase",
                "Unknown"
            )

            context_parts.append(
                f"""
IMAGE SOURCE {image_number}

Image ID:
{image_id}

Image filename:
{image_filename}

Construction phase:
{construction_phase}

Semantic distance:
{distance:.4f}

Image analysis:
{document}
"""
            )


        # ====================================================
        # PDF
        # ====================================================

        else:

            pdf_number += 1

            source = metadata.get(
                "source",
                "Unknown PDF"
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            chunk = metadata.get(
                "chunk",
                "Unknown"
            )

            context_parts.append(
                f"""
PDF SOURCE {pdf_number}

Source:
{source}

Page:
{page}

Chunk:
{chunk}

Semantic distance:
{distance:.4f}

Document text:
{document}
"""
            )


    if not context_parts:

        return (
            "No sufficiently relevant evidence "
            "was retrieved."
        )


    return "\n".join(
        context_parts
    )


# ============================================================
# 9. BUILD RAG PROMPT
# ============================================================

def build_prompt(
    question,
    context,
    question_type
):

    prompt = f"""
You are a Construction Intelligence Assistant.

Your task is to answer the user's question using ONLY
the retrieved evidence provided below.

USER QUESTION:
{question}

QUESTION TYPE:
{question_type}


RETRIEVED EVIDENCE:
{context}


IMPORTANT RULES:

1. Do not invent facts.

2. Do not use outside knowledge when the retrieved
   evidence does not support the answer.

3. For an IMAGE question:
   Use only facts from the image analysis.

4. For a PDF question:
   Use only facts from the PDF/document evidence.

5. For a MIXED question:
   Clearly separate:
   A. What is observed in the construction image.
   B. What the safety/document evidence recommends.

6. Never claim that a PDF recommendation was
   visually observed in the image.

7. Never claim that an image observation was stated
   in the PDF unless the PDF evidence explicitly says so.

8. If the evidence is insufficient, say:
   "I could not find enough information in the provided evidence."

9. Give a clear and concise answer.

10. Use bullet points where useful.

11. Mention the relevant source/page or image when
    appropriate.

12. Do not mention this prompt or these instructions
    in your answer.


ANSWER:
"""

    return prompt


# ============================================================
# 10. GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(prompt):

    try:

        response = gemini_client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt
        )

        # Gemini Interactions API provides the
        # final generated answer through output_text.
        answer = getattr(
            response,
            "output_text",
            None
        )

        if answer:
            return answer.strip()


        # Fallback for different response structures
        if hasattr(response, "outputs"):

            for output in response.outputs:

                text = getattr(
                    output,
                    "text",
                    None
                )

                if text:
                    return text.strip()


        return "Gemini returned no answer."


    except Exception as error:

        return (
            "Gemini answer generation failed.\n\n"
            f"Error: {error}"
        )


# ============================================================
# 11. PRINT RETRIEVED EVIDENCE
# ============================================================

def print_retrieved_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n")
    print("=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)


    if not documents:

        print("\nNo relevant evidence found.")

        return


    image_count = 0
    pdf_count = 0


    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        source_type = get_source_type(
            metadata
        )


        if source_type == "image":

            image_count += 1

            print("\n" + "-" * 70)
            print(
                f"IMAGE RESULT {image_count}"
            )
            print("-" * 70)

            print(
                f"Image: "
                f"{metadata.get('image_filename', 'Unknown')}"
            )

            print(
                f"Image ID: "
                f"{metadata.get('image_id', 'Unknown')}"
            )

            print(
                f"Construction Phase: "
                f"{metadata.get('construction_phase', 'Unknown')}"
            )

            print(
                f"Distance: {distance:.4f}"
            )

            print("\nEvidence:")
            print(document)


        else:

            pdf_count += 1

            print("\n" + "-" * 70)
            print(
                f"PDF RESULT {pdf_count}"
            )
            print("-" * 70)

            print(
                f"PDF: "
                f"{metadata.get('source', 'Unknown')}"
            )

            print(
                f"Page: "
                f"{metadata.get('page', 'Unknown')}"
            )

            print(
                f"Chunk: "
                f"{metadata.get('chunk', 'Unknown')}"
            )

            print(
                f"Distance: {distance:.4f}"
            )

            print("\nEvidence:")
            print(document)


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("CONSTRUCTION INTELLIGENCE COPILOT")
    print("=" * 70)


    # --------------------------------------------------------
    # ChromaDB count
    # --------------------------------------------------------

    print(
        f"\nChromaDB records available: "
        f"{collection.count()}"
    )


    # --------------------------------------------------------
    # Ask question
    # --------------------------------------------------------

    question = input(
        "\nAsk a construction question: "
    ).strip()


    if not question:

        print(
            "\nNo question entered."
        )

        exit()


    # --------------------------------------------------------
    # Detect question type
    # --------------------------------------------------------

    question_type = detect_question_type(
        question
    )

    print(
        f"\nQuestion Type: {question_type}"
    )


    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    print(
        "\nSearching ChromaDB..."
    )

    results = retrieve_documents(
        question
    )


    # --------------------------------------------------------
    # Show retrieved evidence
    # --------------------------------------------------------

    print_retrieved_context(
        results
    )


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        results
    )


    # --------------------------------------------------------
    # Build RAG prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        question,
        context,
        question_type
    )


    # --------------------------------------------------------
    # Generate final answer
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )


    # --------------------------------------------------------
    # Display final answer
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print("\n")
    print(answer)

    print("\n")
    print("=" * 70)
    print("RAG PIPELINE COMPLETE")
    print("=" * 70)