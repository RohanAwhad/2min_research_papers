import asyncio
from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
import time
import os

# Use Agent from pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.exceptions import LLMError, LLMNotAvailableError

from src.pipeline.steps.fetch_arxiv import ArxivPaper
from src.config.settings import settings
from src.utils.logging_config import logger

# Pydantic model for the desired structured summary output
class StructuredSummary(BaseModel):
    problem: str = Field(..., description="What specific problem does the paper address?")
    solution: str = Field(..., description="What is the core proposed solution, method, or approach?")
    results: str = Field(..., description="What are the key findings, results, or conclusions reported? Include limitations if mentioned.")

# Initialize the pydantic-ai Agent
# It will automatically use the GOOGLE_API_KEY from the environment if set
# and the model specified.
llm_agent: Optional[Agent] = None
try:
    # Check if API key is explicitly set in settings (pydantic-ai Agent might use env vars directly)
    if settings.google_api_key:
        # pydantic-ai Agent primarily looks for ENV VARS, but we can set it explicitly
        # This ensures it uses the key from .env if it's there.
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        logger.info("Set GOOGLE_API_KEY environment variable from settings.")

    llm_agent = Agent(settings.llm_model_name)
    logger.info(f"Initialized pydantic-ai Agent with model: {settings.llm_model_name}")
    # Test connectivity/auth if possible (Agent doesn't have a direct ping)
    # We rely on the first call to `run` to detect issues.

except LLMNotAvailableError:
    logger.error(f"LLM model '{settings.llm_model_name}' or GOOGLE_API_KEY not available/configured. Summarization will fail.")
    llm_agent = None # Ensure agent is None if setup fails
except Exception as e:
    logger.error(f"Failed to initialize pydantic-ai Agent: {e}")
    llm_agent = None

async def generate_summary(
    paper: ArxivPaper,
    text_to_summarize: Optional[str],
    source_type: str
) -> Tuple[ArxivPaper, Optional[StructuredSummary], str, Optional[str]]:
    """Generates a structured summary for a single paper using the pydantic-ai Agent."""

    if llm_agent is None:
        logger.error("LLM Agent not initialized. Skipping summarization.")
        return paper, None, source_type, "LLM Agent not initialized"

    if not text_to_summarize:
        logger.warning(f"No text available to summarize for {paper.arxiv_id} (source: {source_type}). Skipping summarization.")
        return paper, None, source_type, "No text provided for summarization"

    # Basic check for text length
    if len(text_to_summarize) < 100:
        logger.warning(f"Text for {paper.arxiv_id} is very short ({len(text_to_summarize)} chars). May result in poor summary. Source: {source_type}")

    # Truncate long texts
    # Gemini 1.5 Pro has a large context, but let's still be mindful.
    # Context window is measured in tokens, not chars, but this is a rough safety net.
    max_input_chars = 1000000 # Increased limit for 1.5 Pro (still approximate)
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
        summary_object = await asyncio.to_thread(
            llm_agent.run, # The synchronous function to run
            prompt,        # The prompt argument
            output_model=StructuredSummary # The output_model argument
        )
        duration = time.time() - start_time
        logger.info(f"Successfully generated summary for {paper.arxiv_id} via Agent in {duration:.2f} seconds.")

    except LLMError as e:
        # Handle specific pydantic-ai LLM errors (includes API errors, validation errors etc.)
        duration = time.time() - start_time
        error_message = f"LLM Agent failed for {paper.arxiv_id} after {duration:.2f}s: {type(e).__name__} - {e}"
        logger.error(error_message)
    except Exception as e:
        # Catch any other unexpected errors during the agent run
        duration = time.time() - start_time
        error_message = f"Unexpected error during Agent run for {paper.arxiv_id} after {duration:.2f}s: {type(e).__name__} - {e}"
        logger.error(error_message)

    return paper, summary_object, source_type, error_message

async def generate_summaries_for_papers(
    extraction_results: List[Tuple[ArxivPaper, Optional[str], str, Optional[str]]]
) -> List[Tuple[ArxivPaper, Optional[StructuredSummary], str, Optional[str]]]:
    """Generates summaries for all papers with extracted text using the Agent."""

    if llm_agent is None:
        logger.error("LLM Agent not available. Cannot generate summaries.")
        # Return original data indicating failure
        return [(paper, None, source, "LLM Agent not available") for paper, _, source, _ in extraction_results]

    # Prepare tasks, but run them sequentially or with controlled concurrency
    # because the Agent runs synchronously within asyncio.to_thread
    results = []
    total_papers = len(extraction_results)
    logger.info(f"Generating summaries for {total_papers} papers sequentially using Agent...")

    for i, (paper, text, source_type, _) in enumerate(extraction_results):
        logger.debug(f"Processing paper {i+1}/{total_papers}: {paper.arxiv_id}")
        try:
            result = await generate_summary(paper, text, source_type)
            results.append(result)
            # Optional: Add a small delay to be kind to APIs, although less critical with sequential execution
            # await asyncio.sleep(0.5)
        except Exception as e:
            # Catch errors from generate_summary if await itself fails (less likely here)
            logger.error(f"Error processing paper {paper.arxiv_id} in main loop: {e}")
            results.append((paper, None, source_type, f"Outer task execution error: {e}"))

        if (i + 1) % 10 == 0:
             logger.info(f"Generated summaries for {i+1}/{total_papers} papers.")

    successful_summaries = sum(1 for _, summary, _, _ in results if summary is not None)
    failed_summaries = total_papers - successful_summaries
    logger.info(f"Summarization summary: Successful={successful_summaries}, Failed={failed_summaries}")

    return results

# Example usage (for testing)
async def main():
    from datetime import date # Use datetime.date for ArxivPaper

    # This test requires actual API calls and assumes GOOGLE_API_KEY is set in the environment
    if not settings.google_api_key and not os.getenv("GOOGLE_API_KEY"):
        print("Skipping summarization test: GOOGLE_API_KEY not set in settings or environment.")
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
        (dummy_paper_1, "This is the full text extracted from the PDF for paper 1. It discusses various approaches to testing AI summarization pipelines, focusing on structured output validation and robustness against noisy input text. The proposed method involves using pydantic-ai with Gemini.", "full_text_pypdf2", None),
        (dummy_paper_2, None, "failed", "PDF download failed")
    ]

    logger.info("Starting summarization test...")
    summary_results = await generate_summaries_for_papers(dummy_extraction_results)

    for paper, summary, source, error in summary_results:
        print(f"\nPaper: {paper.arxiv_id} (Source: {source})")
        if summary:
            print(f"Success:")
            print(json.dumps(summary.model_dump(), indent=2))
        else:
            print(f"Failed: Error: {error}")

if __name__ == "__main__":
    # Ensure .env is loaded if running directly
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())
