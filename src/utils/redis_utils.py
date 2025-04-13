import redis.asyncio as redis # Use asyncio version for potential future async operations
from redis.exceptions import ConnectionError
from src.config.settings import settings
from src.utils.logging_config import logger # Use configured logger

_redis_pool = None

async def get_redis_connection() -> redis.Redis:
    """Establishes and returns an asynchronous Redis connection using a connection pool."""
    global _redis_pool
    if _redis_pool is None:
        try:
            logger.info(f"Connecting to Redis at {settings.redis_host}:{settings.redis_port} DB: {settings.redis_db}")
            # Use decode_responses=True to automatically decode bytes to strings
            _redis_pool = redis.ConnectionPool(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True # Decode responses to UTF-8 strings
            )
            # Test connection
            test_conn = redis.Redis(connection_pool=_redis_pool)
            await test_conn.ping()
            logger.success("Successfully connected to Redis and pinged.")
            await test_conn.close() # Close the temporary test connection

        except ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Potentially raise the error or exit depending on application requirements
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during Redis pool creation: {e}")
            raise

    # Create a client instance from the pool
    return redis.Redis(connection_pool=_redis_pool)

async def close_redis_pool():
    """Closes the Redis connection pool if it exists."""
    global _redis_pool
    if _redis_pool:
        logger.info("Closing Redis connection pool.")
        await _redis_pool.disconnect()
        _redis_pool = None
