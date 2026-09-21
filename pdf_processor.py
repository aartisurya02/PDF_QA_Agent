import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.

    Returns:
        list of dictionaries containing page text and page number.
    """

    # Read PDF using PyPDF2 - it works directly with a file-like object
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pages = []
    for page_number, page in enumerate(pdf_reader.pages, start=1):

        # Extract text from the page object
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    # No explicit close needed for PyPDF2 reader; it will be cleaned up with the file handle

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