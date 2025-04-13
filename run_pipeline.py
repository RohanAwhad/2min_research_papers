import asyncio
import os
import json
from datetime import datetime, timedelta
import arxiv
import pytz
from typing import List, Tuple, Optional

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Configure logging and settings
from src.config.settings import settings
from src.utils.logging_config import logger
from src.utils.redis_utils import close_redis_pool # Only need close pool here

# Import pipeline components
from src.pipeline.steps.fetch_arxiv import ArxivPaper
from src.pipeline.steps.download_pdf import download_all_pdfs, DownloadResult, PDF_STORAGE_PATH
from src.pipeline.steps.extract_text import extract_text_for_papers, ExtractionResult
from src.pipeline.steps.summarize import generate_summaries_for_papers, SummarizationResult, StructuredSummary
from src.pipeline.steps.store_redis import store_results_in_redis


async def fetch_papers(target_date: datetime, categories: List[str]) -> List[ArxivPaper]:
    """Fetches papers from arXiv for a specific date and categories."""
    # Calculate date range
    end_date = target_date.replace(tzinfo=pytz.UTC)
    start_date = end_date - timedelta(days=10)
    
    logger.info(f"Fetching papers between {start_date} and {end_date}")
    
    papers = []
    client = arxiv.Client()

    for category in categories:
        try:
            # Create search query for each category
            search = arxiv.Search(
                query=f'cat:{category}',
                max_results=settings.max_results_per_category,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            # Run client.results in a separate thread as it's synchronous blocking IO
            results = await asyncio.to_thread(lambda: list(client.results(search)))

            # Filter results by date and convert to ArxivPaper objects
            for result in results:
                published_date = result.published.replace(tzinfo=pytz.UTC)
                if start_date <= published_date <= end_date:
                    papers.append(ArxivPaper(
                        entry_id=result.entry_id,
                        arxiv_id=result.get_short_id(),
                        published_date=published_date.date(),
                        title=result.title.strip(),
                        authors=[author.name for author in result.authors],
                        abstract=result.summary.strip(),
                        categories=result.categories,
                        pdf_url=result.pdf_url
                    ))

        except Exception as e:
            logger.error(f"Error fetching papers for category {category}: {e}", exc_info=True)
            continue

    logger.info(f"Fetched {len(papers)} papers published between {start_date.date()} and {end_date.date()}")
    return papers


async def run_pipeline():
    """Runs the full arXiv paper processing pipeline."""
    logger.info("--- Starting arXiv Paper Summarization Pipeline ---")

    target_date = datetime.now(pytz.UTC) - timedelta(days=1)
    logger.info(f"Target Date: {target_date.date().isoformat()}")
    logger.info(f"Target Categories: {settings.arxiv_categories}")
    logger.info(f"LLM Model: {settings.llm_model_name}")
    logger.info(f"Redis Host: {settings.redis_host}:{settings.redis_port}, DB: {settings.redis_db}")
    logger.info(f"PDF Storage Path: {PDF_STORAGE_PATH.resolve()}")

    all_papers: List[ArxivPaper] = []
    download_results: List[DownloadResult] = []
    extraction_results: List[ExtractionResult] = []
    summarization_results: List[SummarizationResult] = []

    try:
        # --- 1. Fetch Papers ---
        logger.info("--- Step 1: Fetching Papers ---")
        all_papers = await fetch_papers(target_date, settings.arxiv_categories)
        if not all_papers:
            logger.warning("No papers found for the target date and categories. Exiting.")
            return

        # --- 2. Download PDFs ---
        logger.info(f"--- Step 2: Downloading PDFs for {len(all_papers)} Papers ---")
        download_results = await download_all_pdfs(all_papers)
        # Filter out papers that failed download for subsequent steps
        successful_downloads = [res for res in download_results if res.file_path is not None]
        failed_downloads = len(download_results) - len(successful_downloads)
        if failed_downloads > 0:
             logger.warning(f"{failed_downloads} paper(s) failed to download.")
        if not successful_downloads:
             logger.warning("No PDFs successfully downloaded. Cannot proceed.")
             return

        # --- 3. Extract Text ---
        logger.info(f"--- Step 3: Extracting Text from {len(successful_downloads)} PDFs ---")
        extraction_results = await extract_text_for_papers(successful_downloads)
        successful_extractions = [res for res in extraction_results if res.text is not None]
        failed_extractions = len(extraction_results) - len(successful_extractions)
        if failed_extractions > 0:
             logger.warning(f"{failed_extractions} paper(s) failed text extraction.")
        if not successful_extractions:
             logger.warning("No text successfully extracted. Cannot proceed.")
             return

        # --- 4. Generate Summaries ---
        logger.info(f"--- Step 4: Generating Summaries for {len(successful_extractions)} Papers ---")
        summarization_results = await generate_summaries_for_papers(successful_extractions)
        successful_summaries = [res for res in summarization_results if res.summary is not None]
        failed_summaries = len(summarization_results) - len(successful_summaries)
        if failed_summaries > 0:
             logger.warning(f"{failed_summaries} paper(s) failed summarization.")
        if not successful_summaries:
             logger.warning("No summaries successfully generated. Nothing to store or print.")
             return

        # --- 5. Store Results ---
        logger.info(f"--- Step 5: Storing {len(successful_summaries)} Results in Redis ---")
        await store_results_in_redis(successful_summaries)
        logger.success("Results stored in Redis.")

        # --- 6. Print Summaries ---
        logger.info("--- Pipeline Finished. Generated Summaries: ---")
        print("\n" + "="*80)
        print(f"Generated Summaries for {target_date.date().isoformat()}")
        print("="*80)
        summary_count = 0
        for paper, summary, source_type, error in summarization_results:
            if summary and not error:
                summary_count += 1
                print(f"\n--- Summary {summary_count} ---")
                print(f"Paper:     {paper.title} ({paper.arxiv_id})")
                print(f"Published: {paper.published_date.isoformat()}")
                print(f"Source:    {source_type}")
                print(f"LLM:       {settings.llm_model_name}")
                print("-" * 20 + " Summary Content " + "-"*20)
                print(f"Problem:\n{summary.problem}\n")
                print(f"Solution:\n{summary.solution}\n")
                print(f"Results:\n{summary.results}\n")
                print("-" * 57) # Match width of content line
            else:
                logger.error(f"Skipping print for {paper.arxiv_id}: Failed at summarization stage. Error: {error}")

        if summary_count == 0:
            print("No summaries were successfully generated in this run.")
        print("="*80)

    except Exception as e:
        logger.error(f"An unexpected error occurred during the pipeline execution: {e}", exc_info=True)

    finally:
        # --- 7. Cleanup ---
        logger.info("--- Cleaning up resources ---")
        # Clean up downloaded PDFs (optional, could be kept for debugging)
        # pdf_files = list(PDF_STORAGE_PATH.glob("*.pdf"))
        # if pdf_files:
        #     logger.info(f"Removing {len(pdf_files)} downloaded PDF files...")
        #     for pdf_file in pdf_files:
        #         try:
        #             os.remove(pdf_file)
        #         except OSError as e:
        #             logger.warning(f"Could not remove PDF file {pdf_file}: {e}")
        # Close Redis pool
        await close_redis_pool()
        logger.info("--- Pipeline Run Complete ---")


if __name__ == "__main__":
    # Ensure API keys are loaded from .env if not set in environment
    if not settings.gemini_api_key and not os.getenv("GEMINI_API_KEY"):
         print("ERROR: GEMINI_API_KEY not found in environment or .env file. Please set it.")
         # Or handle other potential LLM providers based on settings.llm_provider
    elif not settings.arxiv_categories:
         print("ERROR: ARXIV_CATEGORIES not set in environment or .env file. Please set it (e.g., 'cs.LG,cs.CV').")
    else:
        asyncio.run(run_pipeline()) 