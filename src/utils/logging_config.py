import sys
from loguru import logger
from src.config.settings import settings

def setup_logging():
    """Configures Loguru logger based on settings."""
    logger.remove() # Remove default handler
    logger.add(
        sys.stderr, # Log to standard error
        level=settings.log_level.upper(), # Set level from settings
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    logger.info(f"Logging configured with level: {settings.log_level.upper()}")

# Configure logging on import
setup_logging()
