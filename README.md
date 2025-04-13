# arXiv Paper Summarizer

Fetches, summarizes, and stores arXiv papers daily.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```

2.  **Environment Variables:**
    Copy `.env.example` to `.env` and fill in the required values (especially API keys and a secure Redis password).
    ```bash
    cp .env.example .env
    # Edit .env with your values
    ```

3.  **Install Dependencies:**
    Using `uv`:
    ```bash
    uv venv # Create a virtual environment (optional but recommended)
    uv pip install -r pyproject.toml # Use -r for requirements from pyproject.toml
    # or uv sync if you prefer strict lock file usage later
    ```

4.  **Start Redis:**
    Ensure Docker is running and start the Redis container:
    ```bash
    docker-compose up -d
    ```

## Running the Pipeline

```bash
# Ensure your virtual environment is activated if you created one
source .venv/bin/activate # or .env\Scripts\activate on Windows

# Run the main pipeline script
python src/pipeline/main.py
```

## P0 Goals

See `docs/prd.md` for Phase 0 goals and design. 