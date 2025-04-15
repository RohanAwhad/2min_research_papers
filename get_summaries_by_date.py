import asyncio
import sys
import argparse
from datetime import datetime
import json
import time

from src.config.settings import settings
from src.utils.logging_config import logger
from src.utils.redis_utils import get_redis_connection, close_redis_pool
from src.pipeline.steps.summarize import StructuredSummary


async def get_paper_by_id(arxiv_id: str):
    """Retrieves paper data from Redis by arxiv_id."""
    redis_conn = await get_redis_connection()
    paper_key = f"paper:{arxiv_id}"
    paper_data = await redis_conn.hgetall(paper_key)
    return paper_data if paper_data else None


async def get_papers_by_date(date_str):
    """Fetch papers from Redis by date."""
    try:
        # Parse date string
        target_date = datetime.strptime(date_str, "%m/%d/%Y").date()
        date_timestamp = int(datetime(target_date.year, target_date.month, target_date.day).timestamp())

        # Connect to Redis
        redis_conn = await get_redis_connection()

        # Get papers from 'papers_by_date' sorted set within the date range
        next_day_timestamp = date_timestamp + 86400  # Add 24 hours in seconds

        # Get all paper keys with score (publication date) in the range
        paper_keys = await redis_conn.zrangebyscore(
            "papers_by_date",
            date_timestamp,
            next_day_timestamp - 1,
            withscores=True
        )

        if not paper_keys:
            return []

        papers = []
        for paper_key, score in paper_keys:
            # Extract arxiv_id from paper key (format: "paper:{arxiv_id}")
            arxiv_id = paper_key.split(":")[-1] if ":" in paper_key else paper_key

            # Get full paper data
            paper_data = await get_paper_by_id(arxiv_id)
            if paper_data:
                # Get associated summaries
                summary_index_key = f"summaries_for_paper:{arxiv_id}"
                summary_keys = await redis_conn.smembers(summary_index_key)

                # Get the latest summary if available
                summaries = []
                for summary_key in summary_keys:
                    summary_data = await redis_conn.hgetall(summary_key)
                    if summary_data and "summary_content" in summary_data:
                        summary_content = json.loads(summary_data["summary_content"])
                        summary_data["parsed_content"] = summary_content
                        summaries.append(summary_data)

                # Sort summaries by generation timestamp (newest first)
                if summaries:
                    summaries.sort(key=lambda x: int(x.get("generation_timestamp", 0)), reverse=True)

                paper_data["summaries"] = summaries
                papers.append(paper_data)

        return papers

    except Exception as e:
        logger.error(f"Error fetching papers by date: {e}")
        return []


async def main():
    parser = argparse.ArgumentParser(description="Get arXiv paper summaries by date")
    parser.add_argument("date", help="Date in MM/DD/YYYY format")
    args = parser.parse_args()

    try:
        papers = await get_papers_by_date(args.date)

        if not papers:
            print(f"No papers found for date {args.date}")
            return

        print(f"Found {len(papers)} papers for {args.date}")
        print("=" * 80)

        for paper in papers:
            print(f"\nTitle: {paper.get('title', 'Unknown title')}")
            print(f"arXiv ID: {paper.get('arxiv_id', 'Unknown ID')}")
            print(f"Published: {paper.get('published_date', 'Unknown date')}")

            if paper.get("authors"):
                try:
                    authors = json.loads(paper["authors"]) if isinstance(paper["authors"], str) else paper["authors"]
                    author_names = [a.get("name", a) if isinstance(a, dict) else a for a in authors]
                    print(f"Authors: {', '.join(author_names)}")
                except:
                    print(f"Authors: {paper['authors']}")

            if paper.get("categories"):
                try:
                    categories = json.loads(paper["categories"]) if isinstance(paper["categories"], str) else paper["categories"]
                    print(f"Categories: {', '.join(categories)}")
                except:
                    print(f"Categories: {paper['categories']}")

            # Print summaries if available
            if paper.get("summaries"):
                for i, summary in enumerate(paper["summaries"]):
                    print("\n" + "-" * 40 + f" Summary {i+1} " + "-" * 40)
                    print(f"Model used: {summary.get('llm_model_used', 'Unknown model')}")

                    content = summary.get("parsed_content")
                    if content:
                        print("\nPROBLEM:")
                        print(content.get("problem", "Not available"))
                        print("\nSOLUTION:")
                        print(content.get("solution", "Not available"))
                        print("\nRESULTS:")
                        print(content.get("results", "Not available"))
                    else:
                        print("Summary content not available")
            else:
                print("\nNo summary available for this paper")

            print("=" * 80)

    except ValueError:
        print("Error: Date must be in MM/DD/YYYY format")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await close_redis_pool()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_summaries_by_date.py MM/DD/YYYY")
        sys.exit(1)

    asyncio.run(main())
