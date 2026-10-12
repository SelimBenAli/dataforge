"""Small opt-in logging configuration helper."""
import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure basic diagnostic logging for command-line applications."""
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
