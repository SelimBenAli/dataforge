"""DataForge: a local-first data science learning library."""
from .exceptions import AlgorithmError, DataForgeError, DatasetError

__version__ = "0.1.0"
__all__ = ["Dataset", "DataForgeError", "DatasetError", "AlgorithmError", "__version__"]


def __getattr__(name: str):
    """Load the pandas-backed Dataset API only when a user requests it."""
    if name == "Dataset":
        from .data import Dataset
        return Dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
