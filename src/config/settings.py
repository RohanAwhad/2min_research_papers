import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra fields from env file
    )

    # Redis Configuration
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")

    # LLM API Configuration
    # Add fields for your chosen LLM API keys here, e.g.:
    # openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    # anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    llm_model_name: str = Field(default="google-gla:gemini-2.5-pro-exp-03-25", alias="LLM_MODEL_NAME") # Default model

    # Pipeline Configuration
    arxiv_categories_str: str = Field(default="cs.LG,cs.CV", alias="ARXIV_CATEGORIES")
    max_results_per_category: int = Field(default=100, alias="MAX_RESULTS_PER_CATEGORY")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def arxiv_categories(self) -> List[str]:
        """Returns the ARXIV_CATEGORIES string as a list."""
        return [cat.strip() for cat in self.arxiv_categories_str.split(",") if cat.strip()]

# Create a single instance of settings to be imported elsewhere
settings = Settings()
