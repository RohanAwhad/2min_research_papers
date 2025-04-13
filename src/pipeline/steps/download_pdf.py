import asyncio
import os
import requests
from typing import List, Tuple, NamedTuple
from pathlib import Path
import aiohttp
import aiofiles

from src.pipeline.steps.fetch_arxiv import ArxivPaper
from src.utils.logging_config import logger

# Define where PDFs will be stored
PDF_STORAGE_PATH = Path("data/pdfs")
PDF_STORAGE_PATH.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# Define the structure for download results
DownloadResult = NamedTuple("DownloadResult", [
    ("paper", ArxivPaper),
    ("file_path", Path | None),
    ("error", str | None)
])

async def download_pdf(session: aiohttp.ClientSession, paper: ArxivPaper) -> DownloadResult:
    """Downloads a single PDF asynchronously.

    Args:
        session: The aiohttp client session.
        paper: The ArxivPaper object.

    Returns:
        A DownloadResult object containing the paper, file path (or None),
        and error message (or None).
    """
    if not paper.pdf_url:
        return DownloadResult(paper=paper, file_path=None, error="No PDF URL found in metadata.")

    pdf_url_str = str(paper.pdf_url)
    # Ensure the URL ends with .pdf for consistency, some arXiv URLs might not
    if not pdf_url_str.lower().endswith('.pdf'):
        pdf_url_str += '.pdf'

    file_name = f"{paper.arxiv_id.replace('/', '_')}.pdf" # Sanitize filename
    file_path = PDF_STORAGE_PATH / file_name

    # Skip download if file already exists
    if file_path.exists() and file_path.stat().st_size > 0:
        logger.trace(f"PDF already exists: {file_path}")
        return DownloadResult(paper=paper, file_path=file_path, error=None)

    try:
        logger.debug(f"Attempting download: {pdf_url_str}")
        # Use a reasonable timeout
        async with session.get(pdf_url_str, timeout=aiohttp.ClientTimeout(total=60)) as response:
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Check content type if possible, though arXiv usually serves application/pdf
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type:
                logger.warning(f"Unexpected content type '{content_type}' for {pdf_url_str}. Attempting save anyway.")

            # Stream the download to handle potentially large files
            async with aiofiles.open(file_path, mode='wb') as f:
                while True:
                    chunk = await response.content.read(1024 * 1024) # Read 1MB chunks
                    if not chunk:
                        break
                    await f.write(chunk)

            logger.info(f"Successfully downloaded: {file_path}")
            return DownloadResult(paper=paper, file_path=file_path, error=None)

    except aiohttp.ClientResponseError as e:
        error_msg = f"HTTP Error {e.status} for {pdf_url_str}: {e.message}"
        logger.error(error_msg)
        return DownloadResult(paper=paper, file_path=None, error=error_msg)
    except asyncio.TimeoutError:
        error_msg = f"Timeout downloading {pdf_url_str}"
        logger.error(error_msg)
        return DownloadResult(paper=paper, file_path=None, error=error_msg)
    except aiohttp.ClientError as e:
        error_msg = f"Client error downloading {pdf_url_str}: {e}"
        logger.error(error_msg)
        return DownloadResult(paper=paper, file_path=None, error=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error downloading {pdf_url_str}: {type(e).__name__} - {e}"
        logger.error(error_msg)
        # Clean up potentially partially downloaded file on unexpected error
        if file_path.exists():
            try:
                os.remove(file_path)
            except OSError:
                pass
        return DownloadResult(paper=paper, file_path=None, error=error_msg)

async def download_all_pdfs(papers: List[ArxivPaper]) -> List[DownloadResult]:
    """Downloads PDFs for all papers in the list concurrently."""
    if not papers:
        logger.info("No papers provided for PDF download.")
        return []

    async with aiohttp.ClientSession() as session:
        tasks = [download_pdf(session, paper) for paper in papers]
        results: List[DownloadResult] = await asyncio.gather(*tasks)

    successful_downloads = sum(1 for res in results if res.file_path is not None)
    failed_downloads = len(papers) - successful_downloads
    logger.info(f"PDF Download summary: Successful={successful_downloads}, Failed={failed_downloads}")
    return results

# Example usage (for testing)
async def main():
    # Create some dummy paper data
    dummy_papers = [
        ArxivPaper(
            entry_id='http://arxiv.org/abs/2301.00001v1',
            arxiv_id='2301.00001v1',
            published_date=date.today() - timedelta(days=1),
            title="Test Paper 1 (Good Link)",
            authors=["Test Author"],
            abstract="Abstract 1",
            categories=["cs.AI"],
            # pdf_url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf" # Valid dummy PDF
             pdf_url="http://arxiv.org/pdf/2301.00001v1" # Example (may not exist)
        ),
        ArxivPaper(
            entry_id='http://arxiv.org/abs/2301.00002v1',
            arxiv_id='2301.00002v1',
            published_date=date.today() - timedelta(days=1),
            title="Test Paper 2 (Bad Link)",
            authors=["Test Author"],
            abstract="Abstract 2",
            categories=["cs.LG"],
            pdf_url="http://arxiv.org/pdf/invalid-id-99999.pdf"
        ),
         ArxivPaper(
            entry_id='http://arxiv.org/abs/2301.00003v1',
            arxiv_id='2301.00003v1',
            published_date=date.today() - timedelta(days=1),
            title="Test Paper 3 (No Link)",
            authors=["Test Author"],
            abstract="Abstract 3",
            categories=["cs.CV"],
            pdf_url=None
        ),
    ]

    logger.info("Starting dummy PDF download test...")
    results = await download_all_pdfs(dummy_papers)

    for result in results:
        if result.file_path:
            print(f"Success: {result.paper.arxiv_id} -> {result.file_path}")
        else:
            print(f"Failed: {result.paper.arxiv_id} -> Error: {result.error}")

if __name__ == "__main__":
    # Need dummy data imports for testing
    from datetime import date, timedelta
    asyncio.run(main())
