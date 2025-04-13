import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from src.pipeline.steps.fetch_arxiv import ArxivPaper
from src.utils.logging_config import logger

# Define a minimum expected character count for successful full text extraction
# This helps filter out cases where PyPDF2 returns almost nothing.
MIN_EXPECTED_CHARS = 500

async def extract_text_from_pdf(paper: ArxivPaper, pdf_path: Optional[Path]) -> Tuple[ArxivPaper, Optional[str], str, Optional[str]]:
    """Attempts to extract text from a PDF using PyPDF2.

    Args:
        paper: The ArxivPaper object.
        pdf_path: The Path to the downloaded PDF (or None if download failed).

    Returns:
        A tuple containing:
        - The original ArxivPaper object.
        - The extracted text if successful and meets minimum criteria, else None.
        - The source type ("full_text_pypdf2", "abstract", or "failed").
        - An error message if extraction failed, else None.
    """
    if pdf_path is None or not pdf_path.exists():
        logger.warning(f"PDF path not provided or does not exist for {paper.arxiv_id}. Falling back to abstract.")
        return paper, paper.abstract, "abstract", "PDF not available"

    extracted_text = ""
    error_message = None
    source_type = "failed"

    try:
        logger.debug(f"Attempting text extraction from: {pdf_path}")
        reader = await asyncio.to_thread(PdfReader, str(pdf_path)) # Run sync I/O in thread
        num_pages = len(reader.pages)
        logger.trace(f"PDF {pdf_path} has {num_pages} pages.")

        # Extract text page by page
        full_text_list = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = await asyncio.to_thread(page.extract_text) # Sync CPU bound op in thread
                if page_text:
                    full_text_list.append(page_text)
                # else:
                    # logger.trace(f"Page {i+1}/{num_pages} of {pdf_path} had no extractable text.")
            except Exception as page_error:
                logger.warning(f"Error extracting text from page {i+1} of {pdf_path}: {page_error}")
                # Continue to next page

        extracted_text = "\n\n".join(full_text_list).strip()

        if len(extracted_text) >= MIN_EXPECTED_CHARS:
            logger.info(f"Successfully extracted ~{len(extracted_text)} chars from {pdf_path}")
            source_type = "full_text_pypdf2"
        elif paper.abstract: # Check if abstract exists before falling back
            logger.warning(f"Extracted text from {pdf_path} too short ({len(extracted_text)} chars). Falling back to abstract.")
            extracted_text = paper.abstract
            source_type = "abstract"
            error_message = f"PyPDF2 extracted only {len(extracted_text)} chars."
        else:
            logger.error(f"Extracted text from {pdf_path} too short ({len(extracted_text)} chars) and no abstract available.")
            extracted_text = None # No usable text
            source_type = "failed"
            error_message = f"PyPDF2 extraction failed ({len(extracted_text)} chars) and no abstract."

    except PdfReadError as e:
        error_message = f"PyPDF2 error reading {pdf_path}: {e}"
        logger.error(error_message)
        if paper.abstract:
            extracted_text = paper.abstract
            source_type = "abstract"
        else:
             extracted_text = None
             source_type = "failed"
    except Exception as e:
        error_message = f"Unexpected error extracting text from {pdf_path}: {type(e).__name__} - {e}"
        logger.error(error_message)
        if paper.abstract:
            extracted_text = paper.abstract
            source_type = "abstract"
        else:
             extracted_text = None
             source_type = "failed"

    # Final check for None text when source is not 'failed'
    if extracted_text is None and source_type != "failed":
        logger.error(f"Extraction resulted in None text for {paper.arxiv_id} despite source being {source_type}. Setting source to 'failed'.")
        source_type = "failed"
        error_message = error_message or "Final text content was None."

    return paper, extracted_text, source_type, error_message


async def extract_text_for_papers(download_results: List[Tuple[ArxivPaper, Optional[Path], Optional[str]]]) -> List[Tuple[ArxivPaper, Optional[str], str, Optional[str]]]:
    """Extracts text for all papers based on download results."""
    tasks = [extract_text_from_pdf(paper, pdf_path) for paper, pdf_path, _ in download_results]
    results = await asyncio.gather(*tasks)

    successful_extractions = sum(1 for _, text, src, _ in results if text is not None and src == "full_text_pypdf2")
    fallback_abstract = sum(1 for _, _, src, _ in results if src == "abstract")
    failed_extractions = sum(1 for _, _, src, _ in results if src == "failed")

    logger.info(f"Text Extraction summary: Successful (PyPDF2)={successful_extractions}, Fallback (Abstract)={fallback_abstract}, Failed={failed_extractions}")
    return results


# Example usage (for testing)
async def main():
    from src.pipeline.steps.download_pdf import download_all_pdfs # Relative import
    from datetime import date, timedelta

    # Use the dummy papers from download_pdf example
    dummy_papers = [
         ArxivPaper(
            entry_id='http://arxiv.org/abs/2301.00001v1',
            arxiv_id='2301.00001v1',
            published_date=date.today() - timedelta(days=1),
            title="Test Paper 1 (Good Link)",
            authors=["Test Author"],
            abstract="This is abstract 1. It provides a concise summary.",
            categories=["cs.AI"],
            # Use a real PDF known to work with PyPDF2 if possible for better testing
            pdf_url="https://arxiv.org/pdf/2106.07682.pdf" # Example paper that usually works
        ),
        ArxivPaper(
            entry_id='http://arxiv.org/abs/2301.00002v1',
            arxiv_id='2301.00002v1',
            published_date=date.today() - timedelta(days=1),
            title="Test Paper 2 (Bad Link)",
            authors=["Test Author"],
            abstract="This is abstract 2.",
            categories=["cs.LG"],
            pdf_url="http://arxiv.org/pdf/invalid-id-99999.pdf"
        ),
         ArxivPaper(
            entry_id='http://arxiv.org/abs/2301.00003v1',
            arxiv_id='2301.00003v1',
            published_date=date.today() - timedelta(days=1),
            title="Test Paper 3 (No Link)",
            authors=["Test Author"],
            abstract="Abstract 3 is here.",
            categories=["cs.CV"],
            pdf_url=None
        ),
         # Add a case where PDF exists but is hard to parse (if you have an example)
         # ArxivPaper(... pdf_url=... )
    ]

    logger.info("Downloading PDFs for extraction test...")
    download_results = await download_all_pdfs(dummy_papers)

    logger.info("Starting text extraction test...")
    extraction_results = await extract_text_for_papers(download_results)

    for paper, text, source, error in extraction_results:
        status = "Success" if text else "Failed"
        text_preview = (text[:100] + "...") if text else "None"
        print(f"{status}: {paper.arxiv_id} (Source: {source}) -> Preview: '{text_preview}' | Error: {error}")

if __name__ == "__main__":
    asyncio.run(main())
