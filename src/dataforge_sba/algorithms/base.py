"""Algorithm metadata and shared execution interface."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AlgorithmSpec:
    """Stable metadata used by clients to describe a model."""
    id: str
    name: str
    description: str
    category: str
    supervised: bool
    parameters: dict[str, Any] = field(default_factory=dict)


class Algorithm:
    """Base class for registered algorithm implementations."""
    spec: AlgorithmSpec

    def fit_predict(self, X, y=None, **options):
        raise NotImplementedError
