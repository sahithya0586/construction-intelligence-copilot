import re
from collections import Counter
from pdf_loader import load_all_pdfs


# -----------------------------
# Text cleaning
# -----------------------------

def clean_text(text):
    """Clean extracted PDF text."""

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


# -----------------------------
# Find repeated headers/footers
# -----------------------------

def find_repeated_lines(pages):
    """
    Find lines that appear on many pages.
    These are usually PDF headers or footers.
    """

    line_counter = Counter()

    for page in pages:
        text = page["text"]

        for line in text.splitlines():
            line = line.strip()

            if len(line) > 5:
                line_counter[line] += 1

    # Lines appearing on at least 20% of pages
    threshold = max(3, int(len(pages) * 0.2))

    repeated_lines = {
        line for line, count in line_counter.items()
        if count >= threshold
    }

    return repeated_lines


# -----------------------------
# Remove repeated headers
# -----------------------------

def remove_repeated_lines(text, repeated_lines):
    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        if line.strip() not in repeated_lines:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# -----------------------------
# Split large paragraphs
# -----------------------------

def split_large_text(text, max_size=1200):
    """
    Split very large paragraphs into smaller overlapping pieces.
    """

    if len(text) <= max_size:
        return [text]

    chunks = []

    start = 0
    overlap = 200

    while start < len(text):

        end = start + max_size

        # Try to end at a sentence boundary
        if end < len(text):

            sentence_end = max(
                text.rfind(".", start, end),
                text.rfind("?", start, end),
                text.rfind("!", start, end)
            )

            if sentence_end > start + 500:
                end = sentence_end + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# -----------------------------
# Create intelligent chunks
# -----------------------------

def chunk_page(text, max_size=1200):

    text = clean_text(text)

    # Split using paragraph boundaries
    paragraphs = re.split(r"\n\s*\n", text)

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # Handle very large paragraphs
        if len(paragraph) > max_size:

            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            large_chunks = split_large_text(
                paragraph,
                max_size=max_size
            )

            chunks.extend(large_chunks)

            continue

        # Add paragraph to current chunk
        if not current_chunk:

            current_chunk = paragraph

        elif len(current_chunk) + len(paragraph) + 2 <= max_size:

            current_chunk += "\n\n" + paragraph

        else:

            chunks.append(current_chunk.strip())

            # Small overlap from previous chunk
            overlap_text = current_chunk[-200:]

            current_chunk = overlap_text + "\n\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -----------------------------
# Create all chunks
# -----------------------------

def create_chunks():

    documents = load_all_pdfs()

    all_chunks = []

    for document in documents:

        source = document["source"]
        pages = document["pages"]

        # Detect repeated headers/footers
        repeated_lines = find_repeated_lines(pages)

        print(f"\nProcessing: {source}")
        print(f"Repeated lines detected: {len(repeated_lines)}")

        for page in pages:

            page_number = page["page"]

            text = remove_repeated_lines(
                page["text"],
                repeated_lines
            )

            page_chunks = chunk_page(text)

            for chunk_number, chunk in enumerate(
                page_chunks,
                start=1
            ):

                all_chunks.append({
                    "text": chunk,
                    "source": source,
                    "page": page_number,
                    "chunk": chunk_number
                })

    return all_chunks


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":

    chunks = create_chunks()

    print("\n==============================")
    print("IMPROVED CHUNKING TEST")
    print("==============================")

    print(f"Total chunks: {len(chunks)}")

    for item in chunks[:5]:

        print("\n------------------------------")
        print(f"Source: {item['source']}")
        print(f"Page: {item['page']}")
        print(f"Chunk: {item['chunk']}")

        print("\nText:")
        print(item["text"][:800])