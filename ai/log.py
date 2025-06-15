import logging
import sys
from typing import Union

from asgi_correlation_id.context import correlation_id
from loguru import logger

from ai.settings import settings

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} [{correlation_id}] | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "- <level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.

    This handler intercepts all log requests and
    passes them to loguru.

    For more info see:
    https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        """
        Propagates logs to loguru.

        :param record: record to log.
        """
        try:
            level: Union[str, int] = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def _correlation_id_filter(record):
    record["correlation_id"] = correlation_id.get()
    return record["correlation_id"]


def configure_logging() -> None:  # pragma: no cover
    """Configures logging."""
    intercept_handler = InterceptHandler()

    logging.basicConfig(handlers=[intercept_handler], level=logging.NOTSET)

    loggers = [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "openai",
        "httpx",
    ]

    for logger_name in loggers:
        logging.getLogger(logger_name).handlers = []
        logging.getLogger(logger_name).propagate = False
        logging.getLogger(logger_name).addHandler(intercept_handler)

    logging.getLogger("openai").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.INFO)

    # set logs output, level and format
    logger.remove()
    logger.add(
        sink=sys.stdout,
        format=LOG_FORMAT,
        level=settings.log_level.value,
        filter=_correlation_id_filter,
        serialize=False,
    )
