import asyncio
import os
import aiohttp
import time
from datetime import datetime, timedelta
import arxiv
import pytz
from typing import List, Optional, Tuple, Callable, TypeVar, Awaitable, Any
import asyncio.locks

# Configure logging and settings
from src.config.settings import settings
from src.utils.logging_config import logger
from src.utils.redis_utils import close_redis_pool, get_redis_connection

# Import pipeline components
from src.pipeline.steps.fetch_arxiv import ArxivPaper
from src.pipeline.steps.download_pdf import download_pdf, DownloadResult, PDF_STORAGE_PATH
from src.pipeline.steps.extract_text import extract_text_from_pdf, ExtractionResult
from src.pipeline.steps.summarize import generate_summary, SummarizationResult, StructuredSummary
from src.pipeline.steps.store_redis import get_latest_summary, store_results_in_redis, is_paper_data_complete


# Type variable for the rate limiter
T = TypeVar('T')


class LeakyBucketRateLimiter:
    """
    Implements a leaky bucket rate limiter for limiting asynchronous function calls.
    """
    def __init__(self, rate_limit: int, time_period: float = 60.0):
        """
        Initialize the rate limiter.
        
        Args:
            rate_limit: Maximum number of operations allowed in the time period
            time_period: Time period in seconds (default: 60 seconds)
        """
        self.rate_limit = rate_limit  # Operations per time period
        self.time_period = time_period  # Time period in seconds
        self.token_interval = time_period / rate_limit  # Time between tokens
        self.last_token_time = time.time()  # Last time a token was added
        self.tokens = rate_limit  # Start with a full bucket
        self.lock = asyncio.Lock()  # Lock for thread safety
        
        logger.info(f"Rate limiter initialized: {rate_limit} operations per {time_period} seconds "
                   f"(1 operation every {self.token_interval:.2f} seconds)")
    
    async def acquire(self):
        """
        Acquire a token from the bucket, waiting if necessary.
        """
        async with self.lock:
            # Calculate how many tokens should have been added since last check
            current_time = time.time()
            elapsed = current_time - self.last_token_time
            tokens_to_add = elapsed / self.token_interval
            
            # Add tokens to the bucket (up to max capacity)
            self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
            self.last_token_time = current_time
            
            # If we have at least one token, use it immediately
            if self.tokens >= 1:
                self.tokens -= 1
                return
            
            # Otherwise, calculate wait time for next token
            wait_time = self.token_interval - (self.tokens * self.token_interval)
            logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds for next token.")
            
            # Update state as if we had waited
            self.last_token_time += wait_time
            self.tokens = 0  # We'll use the token we're waiting for
            
            # Actually wait
            await asyncio.sleep(wait_time)
    
    async def rate_limited(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute a function with rate limiting.
        
        Args:
            func: The async function to call
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The result of the function call
        """
        await self.acquire()
        return await func(*args, **kwargs)


async def fetch_papers(target_date: datetime, categories: List[str]) -> List[ArxivPaper]:
    """Fetches papers from arXiv for a specific date and categories."""
    # Calculate date range
    end_date: datetime = target_date.replace(tzinfo=pytz.UTC)
    start_date: datetime = end_date - timedelta(days=1)

    logger.info(f"Fetching papers between {start_date} and {end_date}")

    papers: List[ArxivPaper] = []
    client: arxiv.Client = arxiv.Client()

    for category in categories:
        try:
            # Create search query for each category
            search: arxiv.Search = arxiv.Search(
                query=f'cat:{category}',
                max_results=settings.max_results_per_category,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            # Run client.results in a separate thread as it's synchronous blocking IO
            results: List[arxiv.Result] = await asyncio.to_thread(lambda: list(client.results(search)))

            # Filter results by date and convert to ArxivPaper objects
            for result in results:
                published_date: datetime = result.published.replace(tzinfo=pytz.UTC)
                logger.debug(f'Published Date: {published_date}')
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
            logger.exception(f"Error fetching papers for category {category}: {e}")
            continue

    logger.info(f"Fetched {len(papers)} papers published between {start_date.date()} and {end_date.date()}")
    return papers


async def process_summary(paper: ArxivPaper, extraction_result: ExtractionResult) -> Optional[SummarizationResult]:
    """Generates, stores, and prints a summary for a single paper."""
    try:
        has_summary = await is_paper_data_complete(paper.arxiv_id)
        if has_summary:
            logger.info(f'Valid summary found for paper:{paper.arxiv_id}')
            summary_obj = await get_latest_summary(paper.arxiv_id)
            if summary_obj is not None:
                # Print the summary immediately
                print("\n" + "="*80)
                print(f"Retrieved Summary for {paper.published_date.isoformat()}")
                print("="*80)

                print(f"\n--- Summary ---")
                print(f"Paper:     {paper.title} ({paper.arxiv_id})")
                print(f"Published: {paper.published_date.isoformat()}")
                print(f"Source:    {extraction_result.source_type}")
                print(f"LLM:       {settings.llm_model_name}")
                print("-" * 20 + " Summary Content " + "-"*20)
                print(f"Problem:\n{summary_obj.problem}\n")
                print(f"Solution:\n{summary_obj.solution}\n")
                print(f"Results:\n{summary_obj.results}\n")
                print("-" * 57) # Match width of content line
                print("="*80)

                return SummarizationResult(paper, summary_obj, extraction_result.source_type, None)

        summarization_results: List[SummarizationResult] = await generate_summaries_for_papers([extraction_result])

        if not summarization_results or summarization_results[0].summary is None:
            logger.warning(f"Failed to generate summary for {paper.arxiv_id}")
            return None

        summary_result: SummarizationResult = summarization_results[0]
        if summary_result.summary is None: raise Exception('summary is None')

        # Store the result in Redis immediately
        await store_results_in_redis([summary_result])
        logger.info(f"Stored summary for {paper.arxiv_id} in Redis.")

        # Print the summary immediately
        print("\n" + "="*80)
        print(f"Generated Summary for {paper.published_date.isoformat()}")
        print("="*80)

        print(f"\n--- Summary ---")
        print(f"Paper:     {paper.title} ({paper.arxiv_id})")
        print(f"Published: {paper.published_date.isoformat()}")
        print(f"Source:    {summary_result.source_type}")
        print(f"LLM:       {settings.llm_model_name}")
        print("-" * 20 + " Summary Content " + "-"*20)
        print(f"Problem:\n{summary_result.summary.problem}\n")
        print(f"Solution:\n{summary_result.summary.solution}\n")
        print(f"Results:\n{summary_result.summary.results}\n")
        print("-" * 57) # Match width of content line
        print("="*80)

        return summary_result

    except Exception as e:
        logger.error(f"Error processing summary for {paper.arxiv_id}: {e}", exc_info=True)
        return None


async def process_single_paper_summary(paper: ArxivPaper) -> Optional[SummarizationResult]:
    """
    Fully processes a single paper asynchronously:
    1. First checks if a summary already exists in Redis
    2. If no summary exists:
       a. Downloads the PDF if not already downloaded
       b. Extracts text from the PDF
       c. Generates the summary
       d. Stores the summary in Redis
    
    Args:
        paper: The ArxivPaper object containing metadata
        
    Returns:
        Optional[SummarizationResult]: The summarization result, or None if failed
    """
    try:
        # Step 1: Check if we already have a summary in Redis
        has_summary = await is_paper_data_complete(paper.arxiv_id)
        if has_summary:
            logger.info(f'Valid summary found for paper:{paper.arxiv_id}')
            summary_obj = await get_latest_summary(paper.arxiv_id)
            if summary_obj is not None:
                # Construct a SummarizationResult with the existing summary
                # We don't know the source_type for existing summaries, so use 'redis_cache'
                summary_result = SummarizationResult(paper, summary_obj, 'redis_cache', None)
                # Print the summary
                print("\n" + "="*80)
                print(f"Retrieved Summary for {paper.published_date.isoformat()}")
                print("="*80)
                print(f"\n--- Summary ---")
                print(f"Paper:     {paper.title} ({paper.arxiv_id})")
                print(f"Published: {paper.published_date.isoformat()}")
                print(f"Source:    redis_cache") 
                print(f"LLM:       {settings.llm_model_name}")
                print("-" * 20 + " Summary Content " + "-"*20)
                print(f"Problem:\n{summary_obj.problem}\n")
                print(f"Solution:\n{summary_obj.solution}\n")
                print(f"Results:\n{summary_obj.results}\n")
                print("-" * 57) # Match width of content line
                print("="*80)
                return summary_result

        # Step 2: Download PDF if needed (within a ClientSession context)
        async with aiohttp.ClientSession() as session:
            download_result = await download_pdf(session, paper)
        
        if not download_result.file_path:
            logger.warning(f"Failed to download PDF for {paper.arxiv_id}: {download_result.error}")
            # Continue with abstract if available
        
        # Step 3: Extract text from PDF
        extraction_result = await extract_text_from_pdf(paper, download_result.file_path)
        
        if not extraction_result.text:
            logger.warning(f"Failed to extract text for {paper.arxiv_id}: {extraction_result.error}")
            return None
        
        # Step 4: Generate summary
        summarization_result = await generate_summary(paper, extraction_result.text, extraction_result.source_type)
        
        if not summarization_result or summarization_result.summary is None:
            logger.warning(f"Failed to generate summary for {paper.arxiv_id}")
            return None
        
        # Step 5: Store in Redis
        # The function expects a list of tuples: List[Tuple[ArxivPaper, Optional[StructuredSummary], str, Optional[str]]]
        input_format = [(paper, summarization_result.summary, summarization_result.source_type, summarization_result.error)]
        await store_results_in_redis(input_format)
        logger.info(f"Stored summary for {paper.arxiv_id} in Redis.")
        
        # Print the summary
        print("\n" + "="*80)
        print(f"Generated Summary for {paper.published_date.isoformat()}")
        print("="*80)
        print(f"\n--- Summary ---")
        print(f"Paper:     {paper.title} ({paper.arxiv_id})")
        print(f"Published: {paper.published_date.isoformat()}")
        print(f"Source:    {summarization_result.source_type}")
        print(f"LLM:       {settings.llm_model_name}")
        print("-" * 20 + " Summary Content " + "-"*20)
        print(f"Problem:\n{summarization_result.summary.problem}\n")
        print(f"Solution:\n{summarization_result.summary.solution}\n")
        print(f"Results:\n{summarization_result.summary.results}\n")
        print("-" * 57) # Match width of content line
        print("="*80)
        
        return summarization_result
        
    except Exception as e:
        logger.error(f"Error processing summary for {paper.arxiv_id}: {e}", exc_info=True)
        return None

async def run_pipeline():
    """Runs the full arXiv paper processing pipeline."""
    logger.info("--- Starting arXiv Paper Summarization Pipeline ---")

    target_date: datetime = datetime.now(pytz.UTC) - timedelta(days=1)
    logger.info(f"Target Date: {target_date.date().isoformat()}")
    logger.info(f"Target Categories: {settings.arxiv_categories}")
    logger.info(f"LLM Model: {settings.llm_model_name}")
    logger.info(f"Redis Host: {settings.redis_host}:{settings.redis_port}, DB: {settings.redis_db}")
    logger.info(f"PDF Storage Path: {PDF_STORAGE_PATH.resolve()}")

    try:
        # --- 1. Fetch Papers ---
        logger.info("--- Step 1: Fetching Papers ---")
        all_papers = await fetch_papers(target_date, settings.arxiv_categories)
        if not all_papers:
            logger.warning("No papers found for the target date and categories. Exiting.")
            return

        # --- 2. Process Papers with Rate Limiting ---
        # Create a rate limiter allowing 50 operations per minute
        rate_limiter = LeakyBucketRateLimiter(rate_limit=50, time_period=60.0)
        logger.info(f"--- Step 2: Processing {len(all_papers)} Papers with Rate Limiting ({rate_limiter.rate_limit} ops/min) ---")
        
        # Create tasks that will be rate limited
        tasks = []
        for paper in all_papers:
            # Wrap each paper processing call with the rate limiter
            task = rate_limiter.rate_limited(process_single_paper_summary, paper)
            tasks.append(task)
            
        # Execute all tasks and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful summaries
        summary_count = sum(1 for res in results if isinstance(res, SummarizationResult) and res.summary is not None)
        error_count = sum(1 for res in results if isinstance(res, Exception))
        
        logger.info(f"Summary generation results: Success={summary_count}, Errors={error_count}")
        
        if summary_count == 0:
            print("No summaries were successfully generated in this run.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the pipeline execution: {e}", exc_info=True)

    finally:
        # --- 3. Cleanup ---
        logger.info("--- Cleaning up resources ---")
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
