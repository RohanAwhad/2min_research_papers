import asyncio
import time

# Import steps in the correct order
from src.config.settings import settings
from src.utils.logging_config import logger # Initialize logging first
from src.pipeline.steps.fetch_arxiv import fetch_yesterdays_papers
from src.pipeline.steps.download_pdf import download_all_pdfs
from src.pipeline.steps.extract_text import extract_text_for_papers
from src.pipeline.steps.summarize import generate_summaries_for_papers
from src.pipeline.steps.store_redis import store_results_in_redis
from src.utils.redis_utils import close_redis_pool # Import close function

async def run_pipeline():
    """Runs the complete arXiv summarization pipeline."""
    start_time = time.time()
    logger.info("=== Starting arXiv Summarization Pipeline ===")

    try:
        # 1. Fetch paper metadata from arXiv
        logger.info("--- Step 1: Fetching arXiv Papers ---")
        papers = await fetch_yesterdays_papers(
            categories=settings.arxiv_categories,
            max_results=settings.max_results_per_category
        )
        if not papers:
            logger.warning("No papers found for the specified criteria. Pipeline terminating early.")
            return
        logger.info(f"Fetched {len(papers)} papers.")

        # 2. Download PDFs
        logger.info("--- Step 2: Downloading PDFs ---")
        download_results = await download_all_pdfs(papers)
        # download_results is List[Tuple[ArxivPaper, Path | None, str | None]]

        # 3. Extract Text (using PyPDF2)
        logger.info("--- Step 3: Extracting Text (PyPDF2) ---")
        extraction_results = await extract_text_for_papers(download_results)
        # extraction_results is List[Tuple[ArxivPaper, Optional[str], str, Optional[str]]]

        # Filter out papers where text extraction completely failed and no abstract exists
        papers_to_summarize = [
            res for res in extraction_results if res[1] is not None
        ]
        if not papers_to_summarize:
             logger.warning("No papers with available text (full text or abstract) found after extraction. Cannot proceed to summarization.")
             # Still store metadata for fetched papers? Decide based on requirements.
             # For now, we proceed to store metadata even if summaries fail.
        else:
            logger.info(f"{len(papers_to_summarize)} papers have text available for summarization.")


        # 4. Generate Summaries (using LLM)
        logger.info("--- Step 4: Generating Summaries (LLM) ---")
        # Pass only those with text to the summarization step
        summary_results = await generate_summaries_for_papers(papers_to_summarize)
        # summary_results is List[Tuple[ArxivPaper, Optional[StructuredSummary], str, Optional[str]]]

        # Handle papers that failed text extraction (add them back with None summary for storage)
        # Create a map of arxiv_ids that were successfully summarized
        summarized_ids = {paper.arxiv_id for paper, summary, _, _ in summary_results if summary is not None}

        final_results_for_storage = []
        final_results_for_storage.extend(summary_results) # Add successfully processed ones

        # Add papers that failed text extraction or summarization back into the list
        # so their metadata is still stored
        processed_ids = {paper.arxiv_id for paper, _, _, _ in summary_results}
        for paper, extracted_text, source_type, extract_error in extraction_results:
            if paper.arxiv_id not in processed_ids:
                logger.warning(f"Paper {paper.arxiv_id} had text extraction/summarization failure (Source: {source_type}, Error: {extract_error}), adding to storage list with no summary.")
                final_results_for_storage.append((paper, None, source_type, extract_error or "Text extraction failed"))


        # 5. Store Results in Redis
        logger.info("--- Step 5: Storing Results in Redis ---")
        if final_results_for_storage:
            await store_results_in_redis(final_results_for_storage)
        else:
             logger.warning("No final results available to store in Redis.")

    except Exception as e:
        logger.exception("An uncaught error occurred during the pipeline execution.") # Log traceback
    finally:
        # Ensure Redis pool is closed cleanly
        await close_redis_pool()
        end_time = time.time()
        logger.info(f"=== Pipeline Finished in {end_time - start_time:.2f} seconds ===")

if __name__ == "__main__":
    # Ensure .env file is loaded (though settings should handle this)
    from dotenv import load_dotenv
    load_dotenv()

    # Run the async pipeline
    asyncio.run(run_pipeline())
