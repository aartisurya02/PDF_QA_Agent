import fitz
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.

    Returns:
        list of dictionaries containing page text and page number.
    """

    pdf_bytes = pdf_file.read()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    document.close()

    return pages


def create_chunks(pages):
    """
    Split PDF text into smaller chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "page": page["page"]
                }
            )

    return chunks