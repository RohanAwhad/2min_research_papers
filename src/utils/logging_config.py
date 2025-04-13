import base64
import os
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

    try:
        # Langfuse credentials
        LANGFUSE_AUTH = base64.b64encode(f"{settings.LANGFUSE_PUBLIC_KEY}:{settings.LANGFUSE_SECRET_KEY}".encode()).decode()

        # OpenTelemetry endpoints
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:3000/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        trace_provider = TracerProvider()
        trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

        # Sets the global default tracer provider
        from opentelemetry import trace
        trace.set_tracer_provider(trace_provider)

        # OpenLLMetry
        from traceloop.sdk import Traceloop
        Traceloop.init(disable_batch=False,
                       api_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
                       headers=os.environ.get(f"Authorization=Basic {LANGFUSE_AUTH}"),)

        print('logging initialized')
    except Exception as e:
        print(e)
        print('Couldn\'t start logging')


# Configure logging on import
setup_logging()
