from typing import Optional
import pytest
import pytest_asyncio # Import the asyncio fixture decorator
import asyncio
import os
import json
from pathlib import Path
import redis.asyncio as redis
from datetime import date

# Make sure environment is loaded before other imports
from dotenv import load_dotenv
load_dotenv()

# Ensure settings and logger are configured
from src.config.settings import settings
from src.utils.logging_config import logger
from src.utils.redis_utils import get_redis_connection, close_redis_pool
import src.utils.redis_utils # Import the module itself
import src.config.settings # Import the module itself

# Import pipeline steps
import arxiv # Need this for fetching specific ID
from src.pipeline.steps.fetch_arxiv import ArxivPaper
from src.pipeline.steps.download_pdf import download_all_pdfs, PDF_STORAGE_PATH
from src.pipeline.steps.extract_text import extract_text_for_papers
from src.pipeline.steps.summarize import generate_summaries_for_papers, StructuredSummary
from src.pipeline.steps.store_redis import store_results_in_redis

print('='*30, '\nSettings:')
print(settings)
print('='*30)

# Target paper for the test
TEST_ARXIV_ID = "1706.03762" # Attention Is All You Need
TEST_ARXIV_ID_V = "1706.03762v7" # Use a specific version
TEST_REDIS_DB = 1 # Use a dedicated test DB

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# Fixture to manage Redis connection pool - REMOVED as it conflicts with function-scoped one
# @pytest_asyncio.fixture(scope="module", autouse=True)
# async def manage_redis_pool():
#     # Setup: ensure pool is created if not already
#     await get_redis_connection()
#     yield
#     # Teardown: close pool after all tests in module run
#     logger.info("Closing Redis pool after module tests.")
#     await close_redis_pool()

@pytest_asyncio.fixture(scope="function", autouse=True)
async def manage_global_redis_pool():
    """Ensures the global Redis pool is reset for this E2E test."""
    logger.info("Resetting global Redis pool before E2E test.")
    # Ensure the pool is closed if it exists from a previous run/module import
    await close_redis_pool()
    # Reset the global variable to ensure it's created in the test's loop context
    src.utils.redis_utils._redis_pool = None
    # Store original DB setting and override for test
    original_db = src.config.settings.settings.redis_db
    src.config.settings.settings.redis_db = TEST_REDIS_DB

    yield # Run the test

    logger.info("Cleaning up global Redis pool after E2E test.")
    # Clean up the connection pool used by the test
    await close_redis_pool()
    # Reset the global variable again
    src.utils.redis_utils._redis_pool = None
    # Restore original DB setting
    src.config.settings.settings.redis_db = original_db

@pytest_asyncio.fixture # Use pytest_asyncio.fixture
async def redis_cleanup(request):
    """Fixture to clean up Redis data after test execution."""
    created_keys = []
    created_indices = set()

    def add_key(key):
        created_keys.append(key)

    def add_index(index_key):
        created_indices.add(index_key)

    yield add_key, add_index # Pass functions to the test

    # Teardown: Delete keys and index members created during the test
    if not created_keys and not created_indices:
        return

    try:
        redis_conn = await get_redis_connection()
        logger.info(f"Cleaning up Redis keys: {created_keys}")
        if created_keys:
            await redis_conn.delete(*created_keys)

        logger.info(f"Cleaning up Redis indices: {created_indices}")
        paper_key = f"paper:{TEST_ARXIV_ID_V}" # Reconstruct paper key for cleanup
        await redis_conn.zrem("papers_by_date", paper_key)

        categories = ["cs.CL", "cs.LG"]
        for cat in categories:
             cat_key = f"papers_in_category:{cat}"
             if cat_key in created_indices:
                 await redis_conn.srem(cat_key, paper_key)

        summary_index_key = f"summaries_for_paper:{TEST_ARXIV_ID_V}"
        if summary_index_key in created_indices:
             summary_keys = await redis_conn.smembers(summary_index_key)
             if summary_keys:
                 # Ensure we delete the actual summary keys if they weren't added explicitly
                 # This assumes the summary key was added via add_key if it exists
                 pass # Keys should be handled by created_keys delete
             await redis_conn.delete(summary_index_key)

        logger.info("Redis cleanup completed.")

    except Exception as e:
        logger.error(f"Error during Redis cleanup: {e}")

# Check if GEMINI_API_KEY is available, skip test if not
needs_google_key = pytest.mark.skipif(not settings.gemini_api_key and not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")

@needs_google_key
async def test_e2e_arxiv_paper_workflow(redis_cleanup):
    """Tests the full workflow: fetch, download, extract, summarize, store for a specific paper."""
    add_redis_key, add_redis_index = redis_cleanup
    pdf_file_path: Optional[Path] = None

    try:
        # --- 1. Fetch Specific Paper --- 
        logger.info(f"--- Test Step 1: Fetching paper {TEST_ARXIV_ID} ---")
        search = arxiv.Search(id_list=[TEST_ARXIV_ID])
        client = arxiv.Client()
        results = list(await asyncio.to_thread(client.results, search))

        assert len(results) == 1, f"Expected 1 result for ID {TEST_ARXIV_ID}, got {len(results)}"
        result = results[0]

        # Use the specific version ID found (e.g., v7)
        fetched_arxiv_id = result.get_short_id()
        logger.info(f"Found paper version: {fetched_arxiv_id}")

        # Create ArxivPaper object manually for the test
        paper = ArxivPaper(
            entry_id=result.entry_id,
            arxiv_id=fetched_arxiv_id,
            published_date=result.published.date(),
            title=result.title.strip(),
            authors=[author.name for author in result.authors],
            abstract=result.summary.strip(),
            categories=result.categories,
            pdf_url=result.pdf_url
        )
        logger.success(f"Fetched metadata for {paper.arxiv_id}")

        # --- 2. Download PDF --- 
        logger.info(f"--- Test Step 2: Downloading PDF for {paper.arxiv_id} ---")
        download_results = await download_all_pdfs([paper])
        assert len(download_results) == 1
        dl_paper, pdf_file_path, dl_error = download_results[0]

        assert dl_paper.arxiv_id == paper.arxiv_id
        assert pdf_file_path is not None, f"PDF download failed: {dl_error}"
        assert pdf_file_path.exists(), f"Downloaded PDF path does not exist: {pdf_file_path}"
        assert pdf_file_path.stat().st_size > 1000, f"Downloaded PDF seems too small: {pdf_file_path.stat().st_size} bytes"
        logger.success(f"Downloaded PDF to {pdf_file_path}")

        # --- 3. Extract Text --- 
        logger.info(f"--- Test Step 3: Extracting Text from {pdf_file_path} ---")
        extraction_results = await extract_text_for_papers(download_results)
        assert len(extraction_results) == 1
        ex_paper, extracted_text, source_type, ex_error = extraction_results[0]

        assert ex_paper.arxiv_id == paper.arxiv_id
        assert extracted_text is not None, f"Text extraction failed: {ex_error}"
        assert len(extracted_text) > 500, f"Extracted text seems too short ({len(extracted_text)} chars)"
        # For this specific paper, expect PyPDF2 to work reasonably well
        assert source_type == "full_text_pypdf2", f"Expected source type 'full_text_pypdf2', got '{source_type}'"
        logger.success(f"Extracted text ({len(extracted_text)} chars) from source '{source_type}'")

        # --- 4. Generate Summary --- 
        logger.info(f"--- Test Step 4: Generating Summary for {paper.arxiv_id} ---")
        summary_results = await generate_summaries_for_papers(extraction_results)
        assert len(summary_results) == 1
        sum_paper, summary_obj, sum_source_type, sum_error = summary_results[0]

        assert sum_paper.arxiv_id == paper.arxiv_id
        assert summary_obj is not None, f"Summarization failed: {sum_error}"
        assert isinstance(summary_obj, StructuredSummary)
        assert len(summary_obj.problem) > 10, "Summary problem section seems too short"
        assert len(summary_obj.solution) > 10, "Summary solution section seems too short"
        assert len(summary_obj.results) > 10, "Summary results section seems too short"
        logger.success(f"Generated structured summary: {json.dumps(summary_obj.model_dump(), indent=2)}")

        # --- 5. Store in Redis --- 
        logger.info(f"--- Test Step 5: Storing results in Redis for {paper.arxiv_id} ---")
        await store_results_in_redis(summary_results)
        logger.success("Storage function executed.")

        # Add keys/indices to be cleaned up
        paper_key = f"paper:{paper.arxiv_id}"
        add_redis_key(paper_key)
        add_redis_index("papers_by_date") # Assuming it was added here
        for cat in paper.categories:
            add_redis_index(f"papers_in_category:{cat}")
        summary_index_key = f"summaries_for_paper:{paper.arxiv_id}"
        add_redis_index(summary_index_key)
        # We also need to add the summary key itself, but we don't know the UUID... fetch it
        redis_conn_verify = await get_redis_connection()
        summary_keys_in_set = await redis_conn_verify.smembers(summary_index_key)
        assert len(summary_keys_in_set) == 1, f"Expected 1 summary key in index {summary_index_key}, found {len(summary_keys_in_set)}"
        summary_key = list(summary_keys_in_set)[0]
        add_redis_key(summary_key)


        # --- 6. Verify in Redis --- 
        logger.info(f"--- Test Step 6: Verifying data in Redis for {paper.arxiv_id} ---")
        redis_conn_verify = await get_redis_connection()

        # Verify paper data
        stored_paper_data = await redis_conn_verify.hgetall(paper_key)
        assert stored_paper_data is not None, f"Paper key {paper_key} not found in Redis"
        assert len(stored_paper_data) > 5, "Paper hash seems to have too few fields"
        assert stored_paper_data.get("arxiv_id") == paper.arxiv_id
        assert stored_paper_data.get("title") == paper.title
        logger.success(f"Verified paper data for {paper_key}")

        # Verify summary data
        stored_summary_data = await redis_conn_verify.hgetall(summary_key)
        assert stored_summary_data is not None, f"Summary key {summary_key} not found in Redis"
        assert len(stored_summary_data) > 5, "Summary hash seems to have too few fields"
        assert stored_summary_data.get("paper_arxiv_id") == paper.arxiv_id
        assert stored_summary_data.get("llm_model_used") == settings.llm_model_name
        # Verify summary content structure
        summary_content_json = stored_summary_data.get("summary_content")
        assert summary_content_json is not None
        summary_content = json.loads(summary_content_json)
        assert "problem" in summary_content
        assert "solution" in summary_content
        assert "results" in summary_content
        logger.success(f"Verified summary data for {summary_key}")

        # Verify indexing (basic checks)
        assert await redis_conn_verify.zscore("papers_by_date", paper_key) is not None
        assert await redis_conn_verify.sismember(f"papers_in_category:{paper.categories[0]}", paper_key)
        assert await redis_conn_verify.sismember(summary_index_key, summary_key)
        logger.success(f"Verified basic indexing for {paper.arxiv_id}")


    finally:
        # --- 7. Cleanup PDF --- 
        if pdf_file_path and pdf_file_path.exists():
            logger.info(f"--- Test Cleanup: Removing downloaded PDF {pdf_file_path} ---")
            try:
                os.remove(pdf_file_path)
                logger.info("Removed PDF file.")
            except OSError as e:
                logger.warning(f"Could not remove PDF file {pdf_file_path}: {e}")
        # Redis cleanup happens via the fixture
