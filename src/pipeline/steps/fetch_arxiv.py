import arxiv
import asyncio
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, HttpUrl
import json

from src.config.settings import settings
from src.utils.logging_config import logger

# Pydantic model for ArXiv paper metadata (subset)
class ArxivPaper(BaseModel):
    entry_id: str = Field(..., description="The arXiv ID, e.g., 'http://arxiv.org/abs/2301.12345v1'")
    arxiv_id: str = Field(..., description="The core arXiv ID, e.g., '2301.12345v1'")
    published_date: date = Field(..., description="Publication date")
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    pdf_url: Optional[HttpUrl] = Field(default=None, description="Direct PDF link")

async def fetch_yesterdays_papers(categories: List[str], max_results: int) -> List[ArxivPaper]:
    """Fetches metadata for papers published yesterday in the specified categories."""
    yesterday = date.today() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y%m%d") # arXiv API format YYYYMMDD
    logger.info(f"Fetching papers from {yesterday_str} for categories: {categories}")

    query = f"cat:({' OR '.join(categories)}) AND submittedDate:[{yesterday_str} TO {yesterday_str}]"
    logger.debug(f"arXiv query: {query}")

    search = arxiv.Search(
        query=query,
        max_results=max_results * len(categories), # Fetch more initially, filter later if needed
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client()
    results = list(await asyncio.to_thread(client.results, search)) # Run sync library in thread

    papers: List[ArxivPaper] = []
    processed_ids = set()

    logger.info(f"Found {len(results)} raw results from arXiv API.")

    for result in results:
        # Ensure we only process papers *exactly* from yesterday
        # The API might sometimes include papers from the day before/after the range edge
        # Also handles timezone differences potentially
        if result.published.date() != yesterday:
            # logger.trace(f"Skipping {result.entry_id} - published date {result.published.date()} != {yesterday}")
            continue

        # Deduplicate based on core arXiv ID (ignore versions for initial fetch)
        core_id = result.get_short_id()
        if core_id in processed_ids:
            # logger.trace(f"Skipping duplicate core ID: {core_id}")
            continue
        processed_ids.add(core_id)

        # Check if it matches *any* of our target categories
        paper_categories = set(result.categories)
        if not paper_categories.intersection(set(categories)):
            # logger.trace(f"Skipping {core_id} - categories {paper_categories} not in target {categories}")
            continue

        try:
            paper = ArxivPaper(
                entry_id=result.entry_id,
                arxiv_id=core_id, # Use the short ID (e.g., 2301.12345v1)
                published_date=result.published.date(),
                title=result.title.strip(),
                authors=[author.name for author in result.authors],
                abstract=result.summary.strip(),
                categories=result.categories,
                pdf_url=result.pdf_url
            )
            papers.append(paper)
        except Exception as e:
            logger.warning(f"Failed to parse paper {result.entry_id}: {e}")
            # Optionally log more details about the failed paper
            # logger.debug(f"Failed paper details: {result}")

    logger.success(f"Successfully fetched and parsed {len(papers)} papers from yesterday ({yesterday_str}).")
    return papers

# Example usage (for testing)
async def main():
    test_categories = settings.arxiv_categories
    max_res = settings.max_results_per_category
    papers = await fetch_yesterdays_papers(test_categories, max_res)
    print(f"Fetched {len(papers)} papers.")
    if papers:
        print("\nExample Paper:")
        print(json.dumps(papers[0].model_dump(mode='json'), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
