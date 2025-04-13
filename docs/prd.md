# PRD & Design Document: arXiv Paper Summarizer

**Version:** 0.2
**Status:** Draft

## 1. Introduction & Overview

**1.1. Problem:** Researchers, students, and practitioners across scientific fields (Computer Science, Machine Learning, Biology, Computational Biology, etc.) face information overload due to the high volume of research papers published daily on platforms like arXiv. Staying current is time-consuming and difficult.

**1.2. Solution:** This project aims to build an automated system that fetches newly published papers from specified arXiv categories, extracts their content, generates concise, structured summaries using Large Language Models (LLMs), and stores them in a **Redis** database for potential downstream use.

**1.3. Goal:** To provide users with easily digestible, structured summaries (~2-minute read) of relevant daily arXiv papers, highlighting the core problem, proposed solution, and key results.

## 2. Goals (P0 - Phase 0)

*   Automatically fetch metadata and PDF links for papers published "yesterday" from specified arXiv categories via the arXiv API.
*   Download the PDF for each identified paper.
*   Extract plain text content from the downloaded PDFs (using PyPDF2 for P0).
*   Generate a structured summary for each paper using a designated LLM (e.g., Gemini 2.5 Pro, Claude 3 Sonnet). The summary must contain distinct sections for:
    *   The problem the paper addresses.
    *   The proposed solution/methodology.
    *   The key results reported (including potential limitations or "bad" results if discernible).
*   Store paper metadata and the generated structured summaries in a **Redis** database, configured for persistence.
*   The entire process should be runnable as an automated pipeline (e.g., daily execution).

## 3. Non-Goals (P0 - Phase 0)

*   **No Web UI/Frontend:** This phase focuses solely on the backend data processing pipeline.
*   **No Video Generation:** AI-generated video summaries are out of scope for P0.
*   **No Advanced PDF Parsing:** Using more robust tools like GROBID is deferred to a later phase. P0 relies on PyPDF2.
*   **No Manual Curation/Editing:** Summaries are purely AI-generated and stored as-is.
*   **No Complex Audience Tailoring:** Summaries are generated generically; targeting specific audiences is handled by *filtering* which papers/summaries are presented to them downstream, not by changing the summary *content* itself.
*   **No Sophisticated Accuracy Evaluation Framework:** Basic checks and reliance on LLM capabilities are assumed for P0. Formal evaluation is a future consideration.
*   **No Complex Relational Queries:** Redis is chosen; complex joins or transactions typical of SQL databases are not expected in P0.

## 4. Target Audience (for downstream use)

*   Researchers, engineers, students, and practitioners in fields covered by arXiv categories (e.g., Computer Science, ML, Biology, Computational Biology).

## 5. Functional Requirements (P0)

*   **FR1:** The system **shall** connect to the arXiv API to retrieve paper metadata (arXiv ID, title, authors, abstract, categories, publication date, PDF link) for specified categories published within the last 24 hours.
*   **FR2:** The system **shall** download the PDF source file for each retrieved paper.
*   **FR3:** The system **shall** attempt to extract plain text content from the downloaded PDF using the PyPDF2 library. Robust error handling for PDF parsing failures is required.
*   **FR4:** The system **shall** submit the extracted text (or potentially just the abstract as a fallback if text extraction fails badly) to a configured LLM API endpoint.
*   **FR5:** The system **shall** prompt the LLM to generate a summary adhering to a specific structure: `{"problem": "...", "solution": "...", "results": "..."}`.
*   **FR6:** The system **shall** store the original paper metadata (from FR1) in the **Redis** database using appropriate keys and data structures.
*   **FR7:** The system **shall** store the generated structured summary (from FR5), along with metadata (LLM used, timestamp, associated paper ID), in the **Redis** database, linked to the paper data.
*   **FR8:** The system **shall** handle potential API errors (arXiv, LLM, Redis) gracefully (e.g., logging, retries where appropriate).
*   **FR9:** The system **shall** be configurable regarding which arXiv categories to monitor.
*   **FR10:** The **Redis** instance **shall** be configured for persistence (e.g., AOF or RDB snapshots) using Docker Compose volumes.

## 6. System Design (P0)

**6.1. Architecture:**
A linear pipeline orchestrated by a Python script/application, potentially triggered by a scheduler (like `cron`). Data is stored in Redis.


+--------------+ +--------------+ +-----------------+ +-------------+ +-------------+
| arXiv API | ---> | PDF | ---> | Text Extractor | ---> | LLM | ---> | Redis |
| Metadata | | Downloader | | (PyPDF2 - P0) | | Summarizer | | Database |
| Fetcher | +--------------+ +-----------------+ +-------------+ +-------------+
+--------------+ | Fallback: Abstract ^
| +------------------------------------+
| (Store Metadata & Summary)
+----------------------------------------------------------------------------------->

**6.2. Data Acquisition:**
*   **Tool:** Python `arxiv` library.
*   **Process:** Query API daily for papers submitted within a specific date range (e.g., yesterday) for a pre-defined list of categories. Retrieve metadata and PDF URLs.
*   **Considerations:** Respect arXiv API usage policies.

**6.3. Text Extraction (P0):**
*   **Tool:** Python `PyPDF2` library.
*   **Process:** Iterate through downloaded PDFs. Use `PyPDF2` to extract text. Handle errors and potentially fall back to abstract.
*   **Design Decision (P0):** PyPDF2 for simplicity, accepting limitations.
*   **Challenges/Risks:** High risk of poor extraction quality. Plan for fallback and P1 improvement (GROBID).

**6.4. Summarization:**
*   **Tool:** LLM API (e.g., Gemini 1.5 Pro, Claude 3 Sonnet) via Python clients. Libraries like `pydantic-ai` or `instructor` for structured output.
*   **Process:** Prompt LLM for JSON output (`problem`, `solution`, `results`). Parse and validate the structure.
*   **Considerations:** Prompt engineering, API costs, error handling.

**6.5. Data Storage:**
*   **Tool:** **Redis** (managed via Docker Compose with a persistent volume).
*   **Strategy:** Utilize Redis Hashes and potentially Sets/Sorted Sets for indexing.
    *   **Papers:** Stored as Hashes.
        *   Key: `paper:<arxiv_id>` (e.g., `paper:2301.12345v1`)
        *   Fields:
            *   `arxiv_id`: `2301.12345v1` (String)
            *   `title`: "Paper Title Here" (String)
            *   `authors`: JSON String (e.g., `'[{"name": "Author One"}, {"name": "Author Two"}]'`)
            *   `abstract`: "Abstract text..." (String)
            *   `categories`: JSON String (e.g., `'["cs.CV", "cs.LG"]'`)
            *   `published_date`: "YYYY-MM-DD" (String)
            *   `pdf_url`: "http://arxiv.org/pdf/..." (String)
            *   `created_at_ts`: Unix Timestamp (Integer/String)
    *   **Summaries:** Stored as Hashes. A unique ID for each summary generation might be useful.
        *   Key: `summary:<unique_summary_id>` (e.g., `summary:uuid4()`) or potentially `summary:<arxiv_id>:<llm_model_used>` if only one per paper/model is expected initially. Let's assume `summary:<unique_summary_id>`.
        *   Fields:
            *   `summary_id`: `<unique_summary_id>` (String)
            *   `paper_arxiv_id`: `2301.12345v1` (String - links back to the paper)
            *   `summary_content`: JSON String (e.g., `'{"problem": "...", "solution": "...", "results": "..."}'`)
            *   `llm_model_used`: "gemini-1.5-pro" (String)
            *   `source_text_type`: "abstract" or "full_text_pypdf2" (String)
            *   `generation_timestamp`: Unix Timestamp (Integer/String)
            *   `is_reviewed`: "0" or "1" (String/Integer)
    *   **Indexing (Manual):** To efficiently retrieve data based on criteria other than the primary key:
        *   **Papers by Date:** Use a Sorted Set `papers_by_date`.
            *   Score: Unix timestamp of `published_date`.
            *   Member: `paper:<arxiv_id>`
        *   **Papers by Category:** Use Sets, one per category `papers_in_category:<category_name>` (e.g., `papers_in_category:cs.CV`).
            *   Member: `paper:<arxiv_id>`
            *   *Management:* Add paper keys to relevant category sets when storing a new paper.
        *   **Summaries per Paper:** Use a Set `summaries_for_paper:<arxiv_id>`.
            *   Member: `summary:<unique_summary_id>`
            *   *Management:* Add summary key to this set when storing a new summary.
*   **Design Decision:** Using Redis simplifies deployment (via Docker Compose) and offers high performance for key-based lookups. The trade-off is the need for manual indexing and less powerful querying compared to SQL. JSON strings store complex data within hash fields.
*   **Persistence:** Redis persistence (AOF or RDB) must be enabled in the Docker Compose configuration and mapped to a host volume to prevent data loss on container restart.

**6.6. Technology Stack (P0):**
*   **Language:** Python 3.x
*   **arXiv API:** `arxiv` library
*   **PDF Download:** `requests`
*   **PDF Parsing:** `PyPDF2`
*   **LLM Interaction:** `google-generativeai`, `anthropic`, etc. Potentially `pydantic-ai` or `instructor`.
*   **Database:** **Redis**
*   **DB Interaction:** `redis-py` (Python client for Redis)
*   **Deployment/Management:** **Docker & Docker Compose** (for running Python app and Redis with persistence).
*   **Scheduling (Optional):** `cron`, `APScheduler`, `Celery`.

## 7. P0 Scope Summary

The initial focus (P0) is to build and validate the core, automated pipeline: Fetch -> Download -> Extract (PyPDF2) -> Summarize (LLM) -> Store (**Redis**). Success means reliably processing daily papers and storing structured summaries in a persistent Redis instance, accepting PyPDF2 limitations.

## 8. Future Considerations (Post-P0 / P1+)

*   **Improve Text Extraction:** Replace PyPDF2 with GROBID.
*   **Video Generation:** Implement AI video generation.
*   **Accuracy Evaluation:** Develop formal evaluation methods.
*   **User Interface/API:** Build user-facing access layer.
*   **Feedback Mechanism:** Allow user feedback on summaries.
*   **Cost Optimization:** Explore LLM cost strategies.
*   **Enhanced Summarization:** Use GROBID sections, multi-stage summarization.
*   **Data Store Migration (Optional):** If querying needs become significantly more complex and relational, evaluate migration to PostgreSQL or another suitable database.
*   **Redis Scaling:** Explore Redis Cluster if load increases significantly.

## 9. Open Questions & Risks

*   **PDF Parsing Reliability (P0):** High risk. (Mitigation: Fallback, logging, plan for P1).
*   **LLM Cost:** Significant cost potential. (Mitigation: Monitor, optimize, consider alternatives).
*   **LLM Output Consistency/Accuracy:** Requires prompt tuning and checks. (Mitigation: Structured output features, spot-checking).
*   **Redis Data Modeling:** Will the chosen Redis structures adequately support future querying needs? (Mitigation: Keep P0 queries simple, re-evaluate for P1 if needed).
*   **Manual Index Management:** Ensuring Redis sets/sorted sets for indexing are correctly maintained requires careful application logic. (Mitigation: Thorough testing of index updates).
*   **Redis Persistence Configuration:** Incorrect volume mapping or persistence settings in Docker Compose could lead to data loss. (Mitigation: Verify configuration, test restart scenarios).
*   **Query Complexity:** Retrieving data based on multiple criteria (e.g., category AND date range) requires more complex application logic (e.g., intersecting sets) compared to SQL. (Mitigation: Accept P0 limitations, optimize specific lookups).