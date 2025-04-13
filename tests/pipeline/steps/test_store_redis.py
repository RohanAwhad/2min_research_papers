import pytest
import pytest_asyncio
import redis.asyncio as redis
import json
import time
from datetime import date
import uuid

from src.pipeline.steps.fetch_arxiv import ArxivPaper
from src.pipeline.steps.summarize import StructuredSummary
from src.pipeline.steps.store_redis import (
    store_paper_metadata,
    store_summary,
    store_results_in_redis,
)
from src.config.settings import settings # Assuming settings can be configured for testing
from src.utils.logging_config import logger # Use logger if needed

# --- Test Configuration ---
# Use a different DB for testing to avoid conflicts
TEST_REDIS_DB = 1
settings.redis_db = TEST_REDIS_DB # Override settings for testing

# --- Fixtures ---

@pytest_asyncio.fixture(scope="function")
async def redis_conn():
    """Provides a Redis connection for testing and cleans up the DB afterwards."""
    pool = None
    conn = None
    try:
        logger.info(f"Setting up Redis test connection to DB {TEST_REDIS_DB}")
        pool = redis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=TEST_REDIS_DB,
            decode_responses=True
        )
        conn = redis.Redis(connection_pool=pool)
        await conn.ping() # Verify connection
        logger.info(f"Flushing Redis DB {TEST_REDIS_DB} before test.")
        await conn.flushdb()
        yield conn # Provide connection to the test
    except redis.exceptions.ConnectionError as e:
        pytest.fail(f"Could not connect to Redis for testing: {e}. Ensure Redis is running.")
    finally:
        if conn:
            logger.info(f"Flushing Redis DB {TEST_REDIS_DB} after test.")
            try:
                await conn.flushdb()
            except Exception as flush_e:
                 logger.error(f"Error flushing test DB: {flush_e}")
            await conn.close()
            logger.trace("Closed test Redis connection.")
        if pool:
            await pool.disconnect()
            logger.trace("Disconnected test Redis pool.")

# --- Test Data Helpers ---

def create_dummy_paper(arxiv_id="2401.00001v1", title="Test Paper", year=2024, month=1, day=1):
    return ArxivPaper(
        entry_id=f'http://arxiv.org/abs/{arxiv_id}',
        arxiv_id=arxiv_id,
        published_date=date(year, month, day),
        title=title,
        authors=["Author One", "Author Two"],
        abstract="This is a test abstract.",
        categories=["cs.AI", "cs.LG"],
        pdf_url=f"http://arxiv.org/pdf/{arxiv_id}.pdf"
    )

def create_dummy_summary(problem="Test problem", solution="Test solution", results="Test results"):
    return StructuredSummary(
        problem=problem,
        solution=solution,
        results=results
    )

# --- Test Cases ---

@pytest.mark.asyncio
async def test_store_paper_metadata_success(redis_conn):
    """Test successfully storing paper metadata and associated indices."""
    paper = create_dummy_paper(arxiv_id="2401.11111v1", year=2024, month=1, day=15)
    key = f"paper:{paper.arxiv_id}"
    date_key = "papers_by_date"
    cat_key_ai = "papers_in_category:cs.AI"
    cat_key_lg = "papers_in_category:cs.LG"

    success, error = await store_paper_metadata(redis_conn, paper)

    assert success
    assert error is None

    # Verify Hash data
    stored_data = await redis_conn.hgetall(key)
    assert stored_data is not None
    assert stored_data.get("arxiv_id") == paper.arxiv_id
    assert stored_data.get("entry_id") == paper.entry_id
    assert stored_data.get("title") == paper.title
    assert json.loads(stored_data.get("authors")) == [{"name": name} for name in paper.authors]
    assert stored_data.get("abstract") == paper.abstract
    assert json.loads(stored_data.get("categories")) == paper.categories
    assert stored_data.get("published_date") == paper.published_date.isoformat()
    assert stored_data.get("pdf_url") == str(paper.pdf_url)
    assert stored_data.get("created_at_ts") is not None # Check presence and type
    assert int(stored_data.get("created_at_ts")) <= int(time.time())

    # Verify Date Index (ZSET)
    date_score = int(time.mktime(paper.published_date.timetuple()))
    score = await redis_conn.zscore(date_key, key)
    assert score is not None
    assert int(score) == date_score

    # Verify Category Index (SET)
    assert await redis_conn.sismember(cat_key_ai, key)
    assert await redis_conn.sismember(cat_key_lg, key)

@pytest.mark.asyncio
async def test_store_paper_metadata_no_pdf_url(redis_conn):
    """Test storing paper metadata when pdf_url is None."""
    paper = create_dummy_paper(arxiv_id="2401.22222v1")
    paper.pdf_url = None # Explicitly set pdf_url to None
    key = f"paper:{paper.arxiv_id}"

    success, error = await store_paper_metadata(redis_conn, paper)

    assert success
    assert error is None

    stored_data = await redis_conn.hgetall(key)
    assert stored_data.get("pdf_url") == "" # Should store empty string

@pytest.mark.asyncio
async def test_store_summary_success(redis_conn):
    """Test successfully storing a summary and its index."""
    paper = create_dummy_paper(arxiv_id="2401.33333v1")
    summary = create_dummy_summary()
    source_type = "abstract"
    paper_summary_index_key = f"summaries_for_paper:{paper.arxiv_id}"

    # Ensure paper exists first (not strictly needed for this func, but good practice)
    await store_paper_metadata(redis_conn, paper)

    success, error = await store_summary(redis_conn, paper, summary, source_type)

    assert success
    assert error is None

    # Verify summary is indexed for the paper
    indexed_summaries = await redis_conn.smembers(paper_summary_index_key)
    assert len(indexed_summaries) == 1
    summary_key = list(indexed_summaries)[0] # Get the stored summary key (e.g., summary:uuid)
    assert summary_key.startswith("summary:")

    # Verify Hash data for the summary
    stored_data = await redis_conn.hgetall(summary_key)
    assert stored_data is not None
    assert stored_data.get("summary_id") is not None
    assert stored_data.get("paper_arxiv_id") == paper.arxiv_id
    stored_summary_obj = StructuredSummary.model_validate_json(stored_data.get("summary_content"))
    assert stored_summary_obj == summary # Compare Pydantic models
    assert stored_data.get("llm_model_used") == settings.llm_model_name
    assert stored_data.get("source_text_type") == source_type
    assert stored_data.get("generation_timestamp") is not None
    assert int(stored_data.get("generation_timestamp")) <= int(time.time())
    assert stored_data.get("is_reviewed") == "0" # Default value

@pytest.mark.asyncio
async def test_store_summary_none(redis_conn):
    """Test that storing a None summary returns False and stores nothing."""
    paper = create_dummy_paper(arxiv_id="2401.44444v1")
    source_type = "full_text"
    paper_summary_index_key = f"summaries_for_paper:{paper.arxiv_id}"

    success, error = await store_summary(redis_conn, paper, None, source_type)

    assert success is False
    assert error == "No summary object generated"

    # Verify no summary key was added to the index
    indexed_summaries = await redis_conn.smembers(paper_summary_index_key)
    assert len(indexed_summaries) == 0

    # Verify no summary hash was created (by checking keys matching pattern)
    # Note: SCAN can be inefficient on large DBs, but fine for testing
    summary_keys = [key async for key in redis_conn.scan_iter("summary:*")]
    # Filter keys potentially created by other tests (using paper ID)
    related_keys = [k for k in summary_keys if paper.arxiv_id in k] # Simple check
    # A more robust check would involve fetching the hash and checking paper_arxiv_id
    found_related = False
    for k in summary_keys:
        h = await redis_conn.hgetall(k)
        if h.get("paper_arxiv_id") == paper.arxiv_id:
            found_related = True
            break
    assert found_related is False

@pytest.mark.asyncio
async def test_store_results_in_redis_mixed(redis_conn):
    """Test storing a list of results with mixed success (paper+summary, paper only)."""
    paper1 = create_dummy_paper(arxiv_id="2401.55555v1", title="Paper with Summary")
    summary1 = create_dummy_summary()
    source1 = "full_text"
    paper2 = create_dummy_paper(arxiv_id="2401.66666v1", title="Paper without Summary")
    source2 = "abstract"

    results = [
        (paper1, summary1, source1, None),
        (paper2, None, source2, "LLM Failed") # Summary is None
    ]

    # Use the actual get_redis_connection temporarily configured for the test DB
    # No need to mock get_redis_connection if settings are overridden globally
    await store_results_in_redis(results)

    # --- Verification for Paper 1 --- 
    key_p1 = f"paper:{paper1.arxiv_id}"
    data_p1 = await redis_conn.hgetall(key_p1)
    assert data_p1.get("title") == paper1.title
    # Verify indices for paper 1
    assert await redis_conn.zscore("papers_by_date", key_p1) is not None
    assert await redis_conn.sismember(f"papers_in_category:{paper1.categories[0]}", key_p1)

    # Verify Summary 1
    summary_index_key_p1 = f"summaries_for_paper:{paper1.arxiv_id}"
    indexed_summaries_p1 = await redis_conn.smembers(summary_index_key_p1)
    assert len(indexed_summaries_p1) == 1
    summary_key_p1 = list(indexed_summaries_p1)[0]
    data_s1 = await redis_conn.hgetall(summary_key_p1)
    assert data_s1 is not None
    assert data_s1.get("paper_arxiv_id") == paper1.arxiv_id
    assert data_s1.get("source_text_type") == source1

    # --- Verification for Paper 2 --- 
    key_p2 = f"paper:{paper2.arxiv_id}"
    data_p2 = await redis_conn.hgetall(key_p2)
    assert data_p2.get("title") == paper2.title
    # Verify indices for paper 2
    assert await redis_conn.zscore("papers_by_date", key_p2) is not None
    assert await redis_conn.sismember(f"papers_in_category:{paper2.categories[0]}", key_p2)

    # Verify NO Summary for Paper 2
    summary_index_key_p2 = f"summaries_for_paper:{paper2.arxiv_id}"
    indexed_summaries_p2 = await redis_conn.smembers(summary_index_key_p2)
    assert len(indexed_summaries_p2) == 0

@pytest.mark.asyncio
async def test_store_results_in_redis_empty(redis_conn, caplog):
    """Test storing an empty list of results."""
    results = []
    await store_results_in_redis(results)

    # Check logs for info message
    # assert any(record.levelname == 'INFO' and "No summarization results to store in Redis." in record.message for record in caplog.records)

    # Verify DB is still empty
    keys = await redis_conn.keys("*")
    logger.debug(f"Keys found in empty test: {keys}") # Add debug log
    assert len(keys) == 0 # Since fixture flushes DB before this test

@pytest.mark.asyncio
async def test_store_results_in_redis_raises_error_on_failure(mocker):
    """Test that store_results_in_redis raises RuntimeError if a sub-call fails."""
    paper1 = create_dummy_paper(arxiv_id="2401.77777v1")
    summary1 = create_dummy_summary()
    paper2 = create_dummy_paper(arxiv_id="2401.88888v1")

    results = [
        (paper1, summary1, "abstract", None),
        (paper2, None, "abstract", None)
    ]

    # Mock get_redis_connection to return a mock connection
    mock_conn = mocker.AsyncMock(spec=redis.Redis)
    # Mock store_paper_metadata to simulate failure for the second paper
    async def mock_store_paper(*args, **kwargs):
        paper = args[1] # Assuming redis_conn is args[0], paper is args[1]
        if paper.arxiv_id == paper2.arxiv_id:
            logger.error("Simulated paper store failure")
            return False, "Simulated DB error during paper store"
        # Simulate success for paper1
        return True, None

    mocker.patch('src.pipeline.steps.store_redis.get_redis_connection', return_value=mock_conn)
    mocker.patch('src.pipeline.steps.store_redis.store_paper_metadata', side_effect=mock_store_paper)
    # We also need to mock store_summary because it will be called for paper1
    mock_store_summary = mocker.patch('src.pipeline.steps.store_redis.store_summary', return_value=(True, None))

    with pytest.raises(RuntimeError) as excinfo:
        await store_results_in_redis(results)

    assert "Redis storage failed" in str(excinfo.value)
    assert "1 paper(s)" in str(excinfo.value) # Failed paper count
    assert "0 summary(s)" in str(excinfo.value) # Successful summary count before failure
    assert "Simulated DB error during paper store" in str(excinfo.value)

    # Ensure store_summary was only called for the paper that didn't fail metadata storage
    assert mock_store_summary.call_count == 1
    mock_store_summary.assert_called_once_with(mock_conn, paper1, summary1, "abstract")

# (Tests will be added here) 