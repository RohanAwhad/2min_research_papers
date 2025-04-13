import asyncio
import json
import time
import uuid
from typing import List, Tuple, Optional
import redis.asyncio as redis

from src.pipeline.steps.fetch_arxiv import ArxivPaper
from src.pipeline.steps.summarize import StructuredSummary
from src.utils.redis_utils import get_redis_connection
from src.utils.logging_config import logger
from src.config.settings import settings

async def store_paper_metadata(redis_conn: redis.Redis, paper: ArxivPaper):
    """Stores the metadata for a single paper in a Redis Hash."""
    key = f"paper:{paper.arxiv_id}"
    paper_data = {
        "arxiv_id": paper.arxiv_id,
        "entry_id": paper.entry_id,
        "title": paper.title,
        "authors": json.dumps([{"name": name} for name in paper.authors]), # Store as JSON string
        "abstract": paper.abstract,
        "categories": json.dumps(paper.categories), # Store as JSON string
        "published_date": paper.published_date.isoformat(),
        "pdf_url": str(paper.pdf_url) if paper.pdf_url else "",
        "created_at_ts": int(time.time()),
    }

    try:
        # Use HSET to set multiple fields at once
        await redis_conn.hset(key, mapping=paper_data)
        logger.trace(f"Stored metadata for paper {key}")

        # --- Indexing --- (As per PRD)
        # Index by date
        date_key = "papers_by_date"
        date_score = int(time.mktime(paper.published_date.timetuple()))
        await redis_conn.zadd(date_key, {key: date_score})
        logger.trace(f"Indexed paper {key} in {date_key}")

        # Index by category - do sequentially to avoid event loop issues
        for category in paper.categories:
            cat_key = f"papers_in_category:{category}"
            await redis_conn.sadd(cat_key, key)
            logger.trace(f"Added paper {key} to category index {cat_key}")

        return True, None
    except Exception as e:
        error_msg = f"Failed to store metadata or indices for paper {key}: {e}"
        logger.error(error_msg)
        return False, error_msg

async def store_summary(redis_conn: redis.Redis, paper: ArxivPaper, summary: Optional[StructuredSummary], source_type: str):
    """Stores the generated summary (or lack thereof) in a Redis Hash."""
    if not summary:
        logger.warning(f"No summary provided for {paper.arxiv_id}, not storing summary object.")
        # Optionally store a record indicating summarization was attempted but failed
        return False, "No summary object generated"

    summary_id = str(uuid.uuid4()) # Generate a unique ID for the summary
    key = f"summary:{summary_id}"

    summary_data = {
        "summary_id": summary_id,
        "paper_arxiv_id": paper.arxiv_id,
        "summary_content": summary.model_dump_json(), # Store the Pydantic model as JSON string
        "llm_model_used": settings.llm_model_name,
        "source_text_type": source_type,
        "generation_timestamp": int(time.time()),
        "is_reviewed": "0", # Default to not reviewed
    }

    try:
        await redis_conn.hset(key, mapping=summary_data)
        logger.trace(f"Stored summary {key} for paper {paper.arxiv_id}")

        # --- Indexing --- (As per PRD)
        # Index summaries for a given paper
        paper_summary_index_key = f"summaries_for_paper:{paper.arxiv_id}"
        await redis_conn.sadd(paper_summary_index_key, key)
        logger.trace(f"Indexed summary {key} in {paper_summary_index_key}")

        return True, None
    except Exception as e:
        error_msg = f"Failed to store summary {key} or index: {e}"
        logger.error(error_msg)
        return False, error_msg


async def store_results_in_redis(
    summary_results: List[Tuple[ArxivPaper, Optional[StructuredSummary], str, Optional[str]]]
):
    """Stores all paper metadata and generated summaries in Redis."""
    if not summary_results:
        logger.info("No summarization results to store in Redis.")
        return

    redis_conn = await get_redis_connection()
    paper_store_tasks = []
    # Store original inputs along with tasks to map results back
    inputs_for_papers = []
    for paper, summary, source_type, _ in summary_results:
        paper_store_tasks.append(store_paper_metadata(redis_conn, paper))
        inputs_for_papers.append((paper, summary, source_type)) # Keep track

    paper_success_count = 0
    paper_fail_count = 0
    errors = []
    successful_paper_inputs = [] # Keep track of inputs for successful papers

    if paper_store_tasks:
        paper_results = await asyncio.gather(*paper_store_tasks, return_exceptions=True) # Catch potential exceptions
        for i, result in enumerate(paper_results):
            paper_input = inputs_for_papers[i] # Get corresponding input
            paper, _, _ = paper_input

            if isinstance(result, Exception):
                paper_fail_count += 1
                error_msg = f"PaperStore Task {paper.arxiv_id} raised: {result}"
                logger.error(error_msg)
                errors.append(error_msg)
            elif isinstance(result, tuple) and len(result) == 2:
                success, error_msg = result
                if success:
                    paper_success_count += 1
                    successful_paper_inputs.append(paper_input) # Add input to list for summary processing
                else:
                    paper_fail_count += 1
                    if error_msg: errors.append(f"PaperStore {paper.arxiv_id}: {error_msg}")
            else:
                # Handle unexpected result format
                paper_fail_count += 1
                error_msg = f"PaperStore Task {paper.arxiv_id} returned unexpected result: {result}"
                logger.error(error_msg)
                errors.append(error_msg)

        logger.info(f"Paper metadata storage: Successful={paper_success_count}, Failed={paper_fail_count}")

    # Now store summaries ONLY for papers that were stored successfully
    summary_store_tasks = []
    summary_success_count = 0
    summary_fail_count = 0
    for paper, summary, source_type in successful_paper_inputs: # Iterate only successful ones
        # Only attempt to store if a summary object actually exists
        if summary:
            summary_store_tasks.append(store_summary(redis_conn, paper, summary, source_type))
        # Else: No summary object was generated OR paper failed, nothing to store

    if summary_store_tasks:
        summary_results_store = await asyncio.gather(*summary_store_tasks, return_exceptions=True) # Catch potential exceptions
        for i, result in enumerate(summary_results_store):
             # Need to know which paper this result corresponds to - requires careful mapping if needed
             # For now, just process success/failure counts
            if isinstance(result, Exception):
                summary_fail_count += 1
                # Ideally, associate error back to specific paper/summary ID if possible
                error_msg = f"SummaryStore Task raised: {result}"
                logger.error(error_msg)
                errors.append(error_msg)
            elif isinstance(result, tuple) and len(result) == 2:
                success, error_msg = result
                if success:
                    summary_success_count += 1
                else:
                    summary_fail_count += 1
                    # Associate error back if possible
                    if error_msg: errors.append(f"SummaryStore: {error_msg}")
            else:
                summary_fail_count += 1
                error_msg = f"SummaryStore Task returned unexpected result: {result}"
                logger.error(error_msg)
                errors.append(error_msg)

        logger.info(f"Summary storage: Successful={summary_success_count}, Failed={summary_fail_count}")

    # --- Check for failures and raise exception ---
    if paper_fail_count > 0 or summary_fail_count > 0:
        logger.debug(f"Raising RuntimeError. Paper Fail: {paper_fail_count}, Summary Fail: {summary_fail_count}") # Add Debug Log
        error_summary = f"Redis storage failed for {paper_fail_count} paper(s) and {summary_fail_count} summary(s). Errors: {'; '.join(errors)}"
        logger.error(error_summary)
        raise RuntimeError(error_summary)

    logger.info("All results stored successfully in Redis.")
    # Note: Redis connection pool is managed globally, no need to close conn here


# Example usage (for testing)
async def main():
    # Create dummy summary results
    dummy_paper_1 = ArxivPaper(
        entry_id='http://arxiv.org/abs/2401.00011v1',
        arxiv_id='2401.00011v1',
        published_date=time.strptime("2024-01-11", "%Y-%m-%d"), # Needs datetime.date
        title="Redis Storage Test Paper",
        authors=["Redis Author", "DB Author"],
        abstract="Testing storage in Redis.",
        categories=["cs.DB", "cs.IR"],
        pdf_url="http://example.com/redis.pdf"
    )
    dummy_summary_1 = StructuredSummary(
        problem="Need to store data.",
        solution="Use Redis hashes and sets.",
        results="Data stored successfully with indices."
    )
    dummy_paper_2 = ArxivPaper(
        entry_id='http://arxiv.org/abs/2401.00012v1',
        arxiv_id='2401.00012v1',
        published_date=time.strptime("2024-01-11", "%Y-%m-%d"), # Needs datetime.date
        title="Failed Summary Test Paper",
        authors=["Error Author"],
        abstract="Testing failed summary storage.",
        categories=["cs.AI"],
        pdf_url="http://example.com/fail.pdf"
    )

    dummy_summary_results_input = [
        (dummy_paper_1, dummy_summary_1, "full_text_pypdf2", None),
        (dummy_paper_2, None, "abstract", "LLM generation failed")
    ]

    # Ensure Redis is running (docker-compose up -d)
    logger.info("Starting Redis storage test...")
    try:
        await store_results_in_redis(dummy_summary_results_input)
        logger.success("Redis storage test completed.")

        # Verification (Optional)
        redis_conn = await get_redis_connection()
        key1 = f"paper:{dummy_paper_1.arxiv_id}"
        key2 = f"paper:{dummy_paper_2.arxiv_id}"
        data1 = await redis_conn.hgetall(key1)
        data2 = await redis_conn.hgetall(key2)
        print(f"\nVerification for {key1}:\n{data1}")
        print(f"\nVerification for {key2}:\n{data2}")

        # Check indices
        cat_key = f"papers_in_category:{dummy_paper_1.categories[0]}"
        members = await redis_conn.smembers(cat_key)
        print(f"\nMembers in {cat_key}: {members}")

        summary_index_key = f"summaries_for_paper:{dummy_paper_1.arxiv_id}"
        summary_keys = await redis_conn.smembers(summary_index_key)
        print(f"\nSummaries indexed for {dummy_paper_1.arxiv_id}: {summary_keys}")
        if summary_keys:
             summary_key = list(summary_keys)[0]
             summary_data = await redis_conn.hgetall(summary_key)
             print(f"  Data for {summary_key}:\n{summary_data}")

    except Exception as e:
        logger.error(f"Redis storage test failed: {e}")
    finally:
        # Clean up dummy data (optional)
        try:
            redis_conn = await get_redis_connection()
            await redis_conn.delete(f"paper:{dummy_paper_1.arxiv_id}", f"paper:{dummy_paper_2.arxiv_id}")
            await redis_conn.zrem("papers_by_date", f"paper:{dummy_paper_1.arxiv_id}", f"paper:{dummy_paper_2.arxiv_id}")
            await redis_conn.delete(f"papers_in_category:{dummy_paper_1.categories[0]}", f"papers_in_category:{dummy_paper_1.categories[1]}")
            await redis_conn.delete(f"papers_in_category:{dummy_paper_2.categories[0]}")
            summary_index_key = f"summaries_for_paper:{dummy_paper_1.arxiv_id}"
            summary_keys = await redis_conn.smembers(summary_index_key)
            if summary_keys:
                await redis_conn.delete(*list(summary_keys))
            await redis_conn.delete(summary_index_key)
            logger.info("Cleaned up dummy Redis data.")
        except Exception as cleanup_e:
            logger.warning(f"Failed to clean up dummy Redis data: {cleanup_e}")
        # Close pool for standalone script run
        from src.utils.redis_utils import close_redis_pool
        await close_redis_pool()

if __name__ == "__main__":
    # Fix date creation for testing
    from datetime import date
    dummy_paper_1.published_date = date(2024, 1, 11)
    dummy_paper_2.published_date = date(2024, 1, 11)
    asyncio.run(main())
