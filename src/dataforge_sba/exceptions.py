"""Clear exceptions used throughout DataForge."""


class DataForgeError(Exception):
    """Base class for expected DataForge errors."""


class DatasetError(DataForgeError):
    """Raised when a dataset cannot be loaded or used."""


class AlgorithmError(DataForgeError):
    """Raised when an algorithm configuration or input is invalid."""
