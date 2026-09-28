from pathlib import Path
from pypdf import PdfReader


# Project root:
# construction-intelligence-copilot/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# PDF folder:
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def load_all_pdfs():
    documents = []

    print(f"Looking for PDFs in: {DOCUMENTS_DIR}")

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"PDF files found: {len(pdf_files)}")

    for pdf_path in pdf_files:
        print(f"\nLoading: {pdf_path.name}")

        pages = load_pdf(pdf_path)

        documents.append({
            "source": pdf_path.name,
            "pages": pages
        })

        print(f"Pages extracted: {len(pages)}")

    return documents


if __name__ == "__main__":

    documents = load_all_pdfs()

    print("\n==============================")
    print("PDF INGESTION TEST")
    print("==============================")

    print(f"Total documents: {len(documents)}")

    for document in documents:
        print(f"\nDocument: {document['source']}")
        print(f"Pages: {len(document['pages'])}")

        if document["pages"]:
            print("\nFirst page preview:")
            print(document["pages"][0]["text"][:1000])