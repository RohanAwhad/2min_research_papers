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

        # Index by category
        index_tasks = []
        for category in paper.categories:
            cat_key = f"papers_in_category:{category}"
            index_tasks.append(redis_conn.sadd(cat_key, key))
            logger.trace(f"Adding paper {key} to category index {cat_key}")
        if index_tasks:
            await asyncio.gather(*index_tasks)
            logger.trace(f"Completed category indexing for paper {key}")

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
    paper_success_count = 0
    paper_fail_count = 0
    summary_success_count = 0
    summary_fail_count = 0

    logger.info(f"Storing {len(summary_results)} results in Redis...")

    # Store papers first, then summaries
    paper_store_tasks = []
    papers_to_summarize = []
    for paper, summary, source_type, _ in summary_results:
        # Store paper metadata regardless of summary success (unless already exists? TBD)
        # For now, let's assume we always try to store/update the paper metadata
        paper_store_tasks.append(store_paper_metadata(redis_conn, paper))
        papers_to_summarize.append((paper, summary, source_type))

    if paper_store_tasks:
        paper_results = await asyncio.gather(*paper_store_tasks)
        paper_success_count = sum(1 for success, _ in paper_results if success)
        paper_fail_count = len(paper_results) - paper_success_count
        logger.info(f"Paper metadata storage: Successful={paper_success_count}, Failed={paper_fail_count}")

    # Now store summaries
    summary_store_tasks = []
    for paper, summary, source_type in papers_to_summarize:
        summary_store_tasks.append(store_summary(redis_conn, paper, summary, source_type))

    if summary_store_tasks:
        summary_results_store = await asyncio.gather(*summary_store_tasks)
        summary_success_count = sum(1 for success, _ in summary_results_store if success)
        summary_fail_count = len(summary_results_store) - summary_success_count
        logger.info(f"Summary storage: Successful={summary_success_count}, Failed={summary_fail_count}")

    # Note: Redis connection pool is managed globally, no need to close conn here
    # await redis_conn.close() # Don't close if using pool


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
