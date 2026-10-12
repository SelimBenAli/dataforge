"""Exploratory analysis and chart-data helpers shared by the UI and notebooks."""
import numpy as np
import pandas as pd
from dataforge_sba.data import Dataset


def summarize(frame):
    """Summarize a DataFrame through the shared Dataset interface."""
    return Dataset(frame).explore()


def chart_data(frame: pd.DataFrame, chart_type: str, columns: list[str], *, limit: int = 20) -> dict:
    """Return JSON-safe values for a selected EDA chart; no plotting dependency required."""
    allowed = {"histogram", "boxplot", "scatter", "correlation", "bar", "frequency"}
    if chart_type not in allowed:
        raise ValueError("Choose histogram, boxplot, scatter, correlation, bar, or frequency.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(map(str, missing))}.")
    if chart_type in {"histogram", "boxplot"}:
        if len(columns) != 1 or not pd.api.types.is_numeric_dtype(frame[columns[0]]):
            raise ValueError("Choose one numeric column for this chart.")
        values = frame[columns[0]].dropna().astype(float).to_numpy()
        if not len(values): raise ValueError("This column has no usable numeric values.")
        if chart_type == "histogram":
            if np.ptp(values) == 0: return {"type": chart_type, "column": columns[0], "labels": [str(values[0])], "values": [len(values)], "message": "All available values are the same."}
            counts, edges = np.histogram(values, bins=min(20, max(5, int(np.sqrt(len(values))))))
            labels = [f"{edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(counts))]
            return {"type": chart_type, "column": columns[0], "labels": labels, "values": counts.tolist()}
        q1, median, q3 = np.quantile(values, [.25, .5, .75])
        return {"type": chart_type, "column": columns[0], "minimum": float(np.min(values)),
                "q1": float(q1), "median": float(median), "q3": float(q3), "maximum": float(np.max(values))}
    if chart_type == "scatter":
        if len(columns) != 2 or any(not pd.api.types.is_numeric_dtype(frame[c]) for c in columns):
            raise ValueError("Choose two numeric columns for a scatterplot.")
        points = frame[columns].dropna().head(2000)
        if points.empty: raise ValueError("These columns have no complete numeric rows to plot.")
        return {"type": chart_type, "x": columns[0], "y": columns[1],
                "points": [{"x": float(x), "y": float(y)} for x, y in points.itertuples(index=False, name=None)]}
    if chart_type == "correlation":
        numeric = frame.select_dtypes(include="number")
        if columns and any(not pd.api.types.is_numeric_dtype(frame[c]) for c in columns):
            raise ValueError("Correlation heatmaps only use numeric columns.")
        if len(columns) >= 2: numeric = numeric[columns]
        elif columns: numeric = numeric[[columns[0]]]
        if numeric.shape[1] < 2: raise ValueError("Choose at least two numeric columns for a correlation heatmap.")
        corr = numeric.corr().fillna(0)
        return {"type": chart_type, "labels": corr.columns.astype(str).tolist(), "matrix": corr.to_numpy().round(4).tolist()}
    if len(columns) != 1 or pd.api.types.is_numeric_dtype(frame[columns[0]]):
        raise ValueError("Choose one categorical column for a bar chart or frequency distribution.")
    all_counts = frame[columns[0]].astype("string").fillna("(missing)").value_counts()
    counts = all_counts.head(limit).copy()
    if len(all_counts) > limit:
        counts.loc["Other categories"] = int(all_counts.iloc[limit:].sum())
    if counts.empty: raise ValueError("This column has no categories to display.")
    return {"type": chart_type, "column": columns[0], "labels": counts.index.astype(str).tolist(),
            "values": counts.astype(int).tolist(), "truncated": len(all_counts) > limit}


def validate_visualization_selection(frame: pd.DataFrame, charts: list[str], columns: list[str] | None = None) -> None:
    """Validate a Notebook Builder chart selection using the same rules as dashboard charts."""
    supported = {"histogram", "boxplot", "scatter", "correlation", "bar", "frequency"}
    if not isinstance(charts, list) or not charts or any(chart not in supported for chart in charts):
        raise ValueError("Choose at least one supported chart type for notebook EDA.")
    columns = columns or []
    unknown = [column for column in columns if column not in frame.columns]
    if unknown: raise ValueError(f"Unknown visualization column(s): {', '.join(map(str, unknown))}.")
    for chart in charts:
        if chart in {"histogram", "boxplot"}:
            source = columns or frame.columns.tolist()
            selected = [c for c in source if pd.api.types.is_numeric_dtype(frame[c])]
            if not selected: raise ValueError(f"{chart.title()} needs a numeric column.")
        elif chart == "scatter":
            source = columns or frame.columns.tolist()
            selected = [c for c in source if pd.api.types.is_numeric_dtype(frame[c])]
            if len(selected) != 2 or any(not pd.api.types.is_numeric_dtype(frame[c]) for c in selected):
                raise ValueError("Scatterplots require exactly two numeric columns.")
        elif chart == "correlation":
            source = columns or frame.columns.tolist()
            selected = [c for c in source if pd.api.types.is_numeric_dtype(frame[c])]
            if len(selected) < 2 or any(not pd.api.types.is_numeric_dtype(frame[c]) for c in selected):
                raise ValueError("Correlation heatmaps require at least two numeric columns.")
        else:
            source = columns or frame.columns.tolist()
            selected = [c for c in source if not pd.api.types.is_numeric_dtype(frame[c])]
            if not selected:
                raise ValueError("Category charts require categorical columns.")
