"""Dataset abstraction with non-destructive inspection."""
from pathlib import Path
from typing import Any

import pandas as pd

from dataforge_sba.exceptions import DatasetError


class Dataset:
    """Load and inspect a local CSV or Excel file without mutating its source."""

    def __init__(self, source: str | Path | pd.DataFrame):
        self.source = str(source) if not isinstance(source, pd.DataFrame) else None
        if isinstance(source, pd.DataFrame):
            self._frame = source.copy(deep=True)
        else:
            self._frame = self._load(Path(source))
        if self._frame.empty or len(self._frame.columns) == 0:
            raise DatasetError("The dataset is empty or has no usable columns.")

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise DatasetError(f"Dataset file not found: {path}")
        ext = path.suffix.lower()
        try:
            if ext == ".csv":
                try:
                    return pd.read_csv(path, sep=None, engine="python")
                except UnicodeDecodeError:
                    return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
            if ext == ".xlsx":
                return pd.read_excel(path)
            if ext == ".xls":
                raise DatasetError(".xls support is not enabled; convert the file to .xlsx.")
            raise DatasetError("Unsupported file type. Choose a CSV or .xlsx file.")
        except DatasetError:
            raise
        except Exception as exc:
            raise DatasetError(f"Could not read dataset: {exc}") from exc

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive copy of the dataset."""
        return self._frame.copy(deep=True)

    @property
    def shape(self) -> tuple[int, int]:
        return self._frame.shape

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self._frame.columns]

    def schema(self) -> list[dict[str, Any]]:
        """Describe columns with inferred type and missing count."""
        result = []
        for name, series in self._frame.items():
            if pd.api.types.is_bool_dtype(series): kind = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series): kind = "datetime"
            elif pd.api.types.is_numeric_dtype(series): kind = "numeric"
            else: kind = "categorical"
            result.append({"name": str(name), "dtype": str(series.dtype), "kind": kind,
                           "missing": int(series.isna().sum()),
                           "missing_percent": round(float(series.isna().mean() * 100), 2),
                           "unique": int(series.nunique(dropna=True))})
        return result

    def preview(self, rows: int = 10) -> list[dict[str, Any]]:
        """Return a JSON-friendly preview."""
        return self._frame.head(rows).astype(object).where(pd.notna(self._frame.head(rows)), None).to_dict("records")

    def explore(self) -> dict[str, Any]:
        """Return a compact data quality and descriptive summary."""
        numeric = self._frame.select_dtypes(include="number")
        categorical = self._frame.select_dtypes(exclude="number")
        return {"rows": self.shape[0], "columns": self.shape[1], "schema": self.schema(),
                "duplicates": int(self._frame.duplicated().sum()),
                "constant_columns": [str(c) for c in self._frame.columns if self._frame[c].nunique(dropna=False) <= 1],
                "numeric_summary": numeric.describe().replace({float("nan"): None}).to_dict(),
                "categorical_summary": {str(c): self._frame[c].value_counts(dropna=False).head(10).to_dict()
                                        for c in categorical.columns},
                "correlations": numeric.corr(numeric_only=True).fillna(0).to_dict()}
