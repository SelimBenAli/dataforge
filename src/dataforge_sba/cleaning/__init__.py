"""Reusable, non-destructive data cleaning helpers."""
import pandas as pd


def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of missing values by column."""
    return pd.DataFrame({"missing": frame.isna().sum(), "percent": frame.isna().mean().mul(100)})


def impute_missing(frame: pd.DataFrame, strategies: dict[str, str] | None = None) -> pd.DataFrame:
    """Return a copy with selected columns imputed using mean, median, or mode."""
    result = frame.copy(deep=True)
    for column, strategy in (strategies or {}).items():
        if column not in result: raise KeyError(f"Unknown column: {column}")
        if strategy == "mean": value = result[column].mean()
        elif strategy == "median": value = result[column].median()
        elif strategy == "mode":
            modes = result[column].mode(dropna=True)
            value = modes.iloc[0] if len(modes) else None
        else: raise ValueError("Strategy must be mean, median, or mode.")
        if value is not None: result[column] = result[column].fillna(value)
    return result


def drop_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with duplicate rows removed."""
    return frame.drop_duplicates().reset_index(drop=True)


def remove_missing(frame: pd.DataFrame, *, axis: str = "rows", columns: list[str] | None = None) -> pd.DataFrame:
    """Remove rows or selected columns containing missing values, returning a copy."""
    if axis not in {"rows", "columns"}:
        raise ValueError("Choose whether to remove rows or columns with missing values.")
    selected = list(columns or frame.columns)
    unknown = [column for column in selected if column not in frame.columns]
    if unknown:
        raise KeyError(f"Unknown column(s): {', '.join(map(str, unknown))}")
    if axis == "rows":
        return frame.dropna(subset=selected).reset_index(drop=True).copy()
    return frame.drop(columns=[column for column in selected if frame[column].isna().any()]).copy()


def model_imputation_strategies(operations: list[dict] | None) -> dict[str, str]:
    """Collect the last selected imputation rule per column for train-only pipelines."""
    strategies = {}
    for operation in operations or []:
        if operation.get("type") == "impute":
            strategies.update(operation.get("strategies") or {})
    return strategies


def iqr_outliers(frame: pd.DataFrame, column: str, factor: float = 1.5) -> pd.Series:
    """Return a Boolean mask identifying IQR outliers in a numeric column."""
    values = frame[column]
    if not pd.api.types.is_numeric_dtype(values): raise TypeError(f"{column} must be numeric")
    q1, q3 = values.quantile([.25, .75]); spread = q3 - q1
    return (values < q1 - factor * spread) | (values > q3 + factor * spread)


def convert_column(frame: pd.DataFrame, column: str, kind: str) -> tuple[pd.DataFrame, int]:
    """Convert a column and report the number of values that became missing."""
    result = frame.copy(deep=True); before = result[column].notna().sum()
    if kind == "numeric": result[column] = pd.to_numeric(result[column], errors="coerce")
    elif kind == "datetime": result[column] = pd.to_datetime(result[column], errors="coerce")
    elif kind == "category": result[column] = result[column].astype("category")
    elif kind == "boolean":
        normalized = result[column].astype("string").str.strip().str.lower()
        mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "y": True, "n": False}
        result[column] = normalized.map(mapping).astype("boolean")
    else: raise ValueError("kind must be numeric, datetime, category, or boolean")
    return result, int(before - result[column].notna().sum())


def encode_categoricals(frame: pd.DataFrame, columns: list[str], method: str = "onehot",
                        category_orders: dict[str, list] | None = None) -> pd.DataFrame:
    """Encode nominal labels one-hot, or ordinal labels using an explicit real order."""
    if method == "onehot": return pd.get_dummies(frame, columns=columns, dtype=int)
    if method == "ordinal":
        result = frame.copy()
        for c in columns:
            categories = (category_orders or {}).get(c)
            if not categories: raise ValueError(f"Provide the real category order for '{c}' before ordinal encoding.")
            order = [str(value) for value in categories]
            observed = set(result[c].dropna().astype(str).unique())
            unknown = observed.difference(order)
            if unknown: raise ValueError(f"Category order for '{c}' is missing: {', '.join(sorted(unknown))}.")
            result[c] = result[c].astype("string").map({value: index for index, value in enumerate(order)}).astype("Int64")
        return result
    raise ValueError("method must be onehot or ordinal")


def apply_operations(frame: pd.DataFrame, operations: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Apply a validated sequence of cleaning operations to a copy of a frame.

    Each result entry describes the observed effect. Input data is never mutated.
    """
    result = frame.copy(deep=True)
    reports = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise ValueError(f"Cleaning operation {index + 1} is invalid.")
        kind = operation.get("type")
        columns = operation.get("columns") or list(result.columns)
        unknown = [column for column in columns if column not in result.columns]
        if unknown:
            raise ValueError(f"Cleaning operation refers to unknown column(s): {', '.join(map(str, unknown))}.")
        before_shape = result.shape
        changed = 0
        if kind in {"impute", "encode", "outliers"} and not operation.get("columns"):
            raise ValueError(f"Choose at least one column for {kind.replace('_', ' ')}.")
        if kind == "impute":
            strategies = operation.get("strategies") or {}
            for column in columns:
                strategy = strategies.get(column)
                if strategy not in {"mean", "median", "mode"}:
                    raise ValueError(f"Choose mean, median, or mode for '{column}'.")
                if strategy in {"mean", "median"} and not pd.api.types.is_numeric_dtype(result[column]):
                    raise ValueError(f"{strategy.title()} imputation needs a numeric column: '{column}'.")
                if result[column].isna().all():
                    raise ValueError(f"'{column}' contains no non-missing values to use for imputation.")
                missing = int(result[column].isna().sum())
                result = impute_missing(result, {column: strategy})
                changed += missing - int(result[column].isna().sum())
            message = f"Filled {changed} missing value(s)."
        elif kind in {"drop_missing_rows", "drop_missing_columns"}:
            result = remove_missing(result, axis="rows" if kind == "drop_missing_rows" else "columns", columns=columns)
            changed = before_shape[0] - result.shape[0] if kind == "drop_missing_rows" else before_shape[1] - result.shape[1]
            message = f"Removed {changed} row(s) with missing values." if kind == "drop_missing_rows" else f"Removed {changed} column(s) containing missing values."
        elif kind == "drop_duplicates":
            result = drop_duplicates(result)
            changed = before_shape[0] - result.shape[0]
            message = f"Removed {changed} duplicate row(s)."
        elif kind == "convert":
            column, target = operation.get("column"), operation.get("target")
            if column not in result.columns:
                raise ValueError("Choose a column to convert.")
            if target not in {"numeric", "datetime", "category", "boolean"}:
                raise ValueError("Choose numeric, datetime, category, or boolean as the new type.")
            result, changed = convert_column(result, column, target)
            message = f"Converted '{column}' to {target}; {changed} non-missing value(s) could not be converted."
        elif kind == "encode":
            method = operation.get("method", "onehot")
            if method not in {"onehot", "ordinal"}:
                raise ValueError("Choose one-hot or ordinal encoding.")
            if method == "ordinal" and not operation.get("ordered"):
                raise ValueError("Ordinal encoding needs confirmation that categories have a real order. Use one-hot for nominal categories.")
            category_order = operation.get("category_order")
            orders = {column: category_order for column in columns} if category_order else None
            result = encode_categoricals(result, columns, method, orders)
            changed = len(columns)
            message = f"{'One-hot' if method == 'onehot' else 'Ordinal'} encoded {changed} categorical column(s)."
        elif kind == "outliers":
            details = []
            for column in columns:
                if not pd.api.types.is_numeric_dtype(result[column]):
                    raise ValueError(f"Outlier detection requires numeric columns: '{column}' is not numeric.")
                mask = iqr_outliers(result, column)
                values = result[column].dropna()
                q1, q3 = values.quantile([.25, .75]) if not values.empty else (None, None)
                details.append({"column": str(column), "count": int(mask.sum()),
                                "q1": float(q1) if pd.notna(q1) else None,
                                "q3": float(q3) if pd.notna(q3) else None})
            changed = sum(item["count"] for item in details)
            message = f"Flagged {changed} potential outlier value(s). No rows were changed."
        else:
            raise ValueError(f"Unknown cleaning operation: {kind}.")
        reports.append({"type": kind, "changed": changed, "details": details if kind == "outliers" else None, "before_rows": before_shape[0],
                        "after_rows": result.shape[0], "before_columns": before_shape[1],
                        "after_columns": result.shape[1], "message": message})
    return result, reports
