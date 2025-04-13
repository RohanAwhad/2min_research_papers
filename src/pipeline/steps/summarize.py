import asyncio
from typing import List, Tuple, Optional, Dict, Any, NamedTuple
from pydantic import BaseModel, Field
import json
import time
import os

# Use Agent from pydantic_ai
from pydantic_ai import Agent

from src.pipeline.steps.fetch_arxiv import ArxivPaper
from src.pipeline.steps.extract_text import ExtractionResult # Import previous step's result type
from src.config.settings import settings
from src.utils.logging_config import logger

# Pydantic model for the desired structured summary output
class StructuredSummary(BaseModel):
    problem: str = Field(..., description="What specific problem does the paper address?")
    solution: str = Field(..., description="What is the core proposed solution, method, or approach?")
    results: str = Field(..., description="What are the key findings, results, or conclusions reported? Include limitations if mentioned.")

# Initialize the pydantic-ai Agent
# llm_agent: Optional[Agent] = None
try:
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
        logger.info("Set GEMINI_API_KEY environment variable from settings.")

    # Attempt to initialize the agent
    llm_agent = Agent(settings.llm_model_name)
    logger.info(f"Initialized pydantic-ai Agent with model: {settings.llm_model_name}")
    # Note: Initialization itself might not fail even if API key is bad.
    # The error might only occur during the first agent.run call.

# Catch any general exception during initialization
except Exception as e:
    logger.exception(f"Failed to initialize pydantic-ai Agent: {type(e).__name__} - {e}")
    logger.exception("LLM features will be unavailable.")
    llm_agent = None # Ensure agent is None if setup fails

SummarizationResult = NamedTuple("SummarizationResult", [
    ("paper", ArxivPaper),
    ("summary", Optional[StructuredSummary]),
    ("source_type", str),
    ("error", Optional[str])
])

async def generate_summary(
    paper: ArxivPaper,
    text_to_summarize: Optional[str],
    source_type: str
) -> SummarizationResult:
    """Generates a structured summary for a single paper using the pydantic-ai Agent."""

    global llm_agent

    if llm_agent is None:
        logger.error("LLM Agent not initialized. Skipping summarization.")
        return SummarizationResult(paper=paper, summary=None, source_type=source_type, error="LLM Agent not initialized")

    if not text_to_summarize:
        logger.warning(f"No text available to summarize for {paper.arxiv_id} (source: {source_type}). Skipping summarization.")
        return SummarizationResult(paper=paper, summary=None, source_type=source_type, error="No text provided for summarization")

    # Basic check for text length
    if len(text_to_summarize) < 100:
        logger.warning(f"Text for {paper.arxiv_id} is very short ({len(text_to_summarize)} chars). May result in poor summary. Source: {source_type}")

    # Truncate long texts
    # Gemini 2.5 Pro has a large context, but let's still be mindful.
    # Context window is measured in tokens, not chars, but this is a rough safety net.
    max_input_chars = 10_000_000 # Increased limit for 2.5 Pro (still approximate)
    if len(text_to_summarize) > max_input_chars:
        logger.warning(f"Text for {paper.arxiv_id} ({len(text_to_summarize)} chars) exceeds rough limit {max_input_chars}. Truncating input.")
        text_to_summarize = text_to_summarize[:max_input_chars]

    # Construct the prompt for the Agent
    # Note: Agent automatically handles asking for the output_model structure.
    prompt = f"""
Analyze the following research paper content and generate a summary.

Paper Title: {paper.title}
Authors: {", ".join(paper.authors)}
Source: '{source_type}' # Information about where the text came from
Abstract (for context): {paper.abstract}

Content to Summarize:
```text
{text_to_summarize}
```

Provide a concise summary focusing on these aspects:
- The specific problem the paper addresses.
- The core proposed solution, method, or approach.
- The key findings, results, or conclusions reported (including limitations if mentioned).

Keep the summary sections direct and informative, suitable for a quick read (~2 minutes).
"""

    summary_object: Optional[StructuredSummary] = None
    error_message: Optional[str] = None
    start_time = time.time()

    try:
        logger.debug(f"Running Agent for {paper.arxiv_id} (text: {len(text_to_summarize)} chars, source: {source_type})")
        # Use the Agent's run method with the desired output model
        # This needs to run in a thread because pydantic-ai's Agent.run is synchronous
        result = await llm_agent.run(prompt, result_type=StructuredSummary)
        summary_object = result.data
        duration = time.time() - start_time
        logger.info(f"Successfully generated summary for {paper.arxiv_id} via Agent in {duration:.2f} seconds.")

    except Exception as e:
        # Catch general exceptions during the agent run instead of specific LLMError
        duration = time.time() - start_time
        error_message = f"LLM Agent run failed for {paper.arxiv_id} after {duration:.2f}s: {type(e).__name__} - {e}"
        logger.error(error_message)
        # Ensure summary_object is None if an error occurred
        summary_object = None

    return SummarizationResult(paper=paper, summary=summary_object, source_type=source_type, error=error_message)

async def generate_summaries_for_papers(
    extraction_results: List[ExtractionResult]
) -> List[SummarizationResult]:
    """Generates summaries for all papers with extracted text using the Agent."""
    global llm_agent

    if llm_agent is None:
        logger.error("LLM Agent not available. Cannot generate summaries.")
        # Return original data indicating failure
        return [SummarizationResult(paper=res.paper, summary=None, source_type=res.source_type, error="LLM Agent not available") for res in extraction_results]

    # Prepare tasks, but run them sequentially or with controlled concurrency
    # because the Agent runs synchronously within asyncio.to_thread
    results: List[SummarizationResult] = []
    total_papers = len(extraction_results)
    logger.info(f"Generating summaries for {total_papers} papers sequentially using Agent...")

    for i, extraction_res in enumerate(extraction_results):
        logger.debug(f"Processing paper {i+1}/{total_papers}: {extraction_res.paper.arxiv_id}")
        try:
            result = await generate_summary(extraction_res.paper, extraction_res.text, extraction_res.source_type)
            results.append(result)
            # Optional: Add a small delay to be kind to APIs, although less critical with sequential execution
            # await asyncio.sleep(0.5)
        except Exception as e:
            # Catch errors from generate_summary if await itself fails (less likely here)
            logger.error(f"Error processing paper {extraction_res.paper.arxiv_id} in main loop: {e}")
            results.append(SummarizationResult(paper=extraction_res.paper, summary=None, source_type=extraction_res.source_type, error=f"Outer task execution error: {e}"))

        if (i + 1) % 10 == 0:
             logger.info(f"Generated summaries for {i+1}/{total_papers} papers.")

    successful_summaries = sum(1 for res in results if res.summary is not None)
    failed_summaries = total_papers - successful_summaries
    logger.info(f"Summarization summary: Successful={successful_summaries}, Failed={failed_summaries}")

    return results

# Example usage (for testing)
async def main():
    from datetime import date # Use datetime.date for ArxivPaper

    # This test requires actual API calls and assumes GEMINI_API_KEY is set in the environment
    if not settings.gemini_api_key and not os.getenv("GEMINI_API_KEY"):
        print("Skipping summarization test: GEMINI_API_KEY not set in settings or environment.")
        return
    if llm_agent is None:
        print("Skipping summarization test: LLM Agent failed to initialize.")
        return

    # Create dummy extraction results
    dummy_paper_1 = ArxivPaper(
        entry_id='http://arxiv.org/abs/2301.00001v1',
        arxiv_id='2301.00001v1',
        published_date=date.today(), # Use date object
        title="Test Paper 1 for Summarization",
        authors=["Test Author"],
        abstract="This abstract describes a novel method for testing summarizers.",
        categories=["cs.AI"],
        pdf_url="http://example.com/dummy.pdf"
    )
    dummy_paper_2 = ArxivPaper(
        entry_id='http://arxiv.org/abs/2301.00002v1',
        arxiv_id='2301.00002v1',
        published_date=date.today(), # Use date object
        title="Test Paper 2 (No Text)",
        authors=["Another Author"],
        abstract="Abstract for paper with no text.",
        categories=["cs.LG"],
        pdf_url=None
    )

    dummy_extraction_results = [
        ExtractionResult(paper=dummy_paper_1, text="This is the full text extracted from the PDF for paper 1. It discusses various approaches to testing AI summarization pipelines, focusing on structured output validation and robustness against noisy input text. The proposed method involves using pydantic-ai with Gemini.", source_type="full_text_pypdf2", error=None),
        ExtractionResult(paper=dummy_paper_2, text=None, source_type="failed", error="PDF download failed")
    ]

    logger.info("Starting summarization test...")
    summary_results = await generate_summaries_for_papers(dummy_extraction_results)

    for result in summary_results:
        print(f"\nPaper: {result.paper.arxiv_id} (Source: {result.source_type})")
        if result.summary:
            print(f"Success:")
            print(json.dumps(result.summary.model_dump(), indent=2))
        else:
            print(f"Failed: Error: {result.error}")

if __name__ == "__main__":
    # Ensure .env is loaded if running directly
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
