"""Central registry of executable scikit-learn algorithms."""
from dataforge_sba.algorithms.base import AlgorithmSpec

SPECS = [
    AlgorithmSpec("linear_regression", "Linear Regression", "Predict a numeric value from features.", "regression", True, {"test_size": .2, "random_state": 42}),
    AlgorithmSpec("logistic_regression", "Logistic Regression", "Classify observations using logistic regression.", "classification", True, {"max_iter": 1000, "test_size": .2, "random_state": 42}),
    AlgorithmSpec("kmeans", "K-means", "Group observations by feature similarity; cluster IDs are not rankings.", "clustering", False, {"n_clusters": 3, "scale": True, "random_state": 42}),
    AlgorithmSpec("decision_tree", "Decision Tree", "Classify data using interpretable decision rules.", "classification", True, {"max_depth": 4, "test_size": .2, "random_state": 42}),
    AlgorithmSpec("knn", "K-Nearest Neighbors", "Classify each point using its nearest training examples.", "classification", True, {"n_neighbors": 5, "test_size": .2, "random_state": 42}),
]


def list_algorithms() -> list[dict]:
    """Return public algorithm metadata."""
    return [vars(s) for s in SPECS]


def _pipeline(model, frame, scale: bool = True, scaler: str = "standard", impute_strategies=None):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
    import pandas as pd
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = frame.columns.difference(numeric).tolist()
    transformers = []
    if scale and scaler not in {"standard", "minmax"}:
        raise ValueError("Choose standard, minmax, or none for feature scaling.")
    strategies = impute_strategies or {}
    for strategy in ("median", "mean", "most_frequent"):
        selected = [column for column in numeric if ("most_frequent" if strategies.get(column) == "mode" else strategies.get(column, "median")) == strategy]
        if selected:
            steps = [("imputer", SimpleImputer(strategy=strategy))]
            if scale: steps.append(("scaler", StandardScaler() if scaler == "standard" else MinMaxScaler()))
            transformers.append((f"numeric_{strategy}", Pipeline(steps), selected))
    if categorical:
        invalid = [column for column in categorical if strategies.get(column) not in (None, "mode", "most_frequent")]
        if invalid: raise ValueError("Categorical features can only use mode imputation.")
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocessing", prep), ("model", model)])


def run_algorithm(identifier: str, frame, *, target: str | None = None,
                  feature_columns: list[str] | None = None, impute_strategies: dict | None = None, **options) -> dict:
    """Fit a registered model and return predictions, metrics, and model metadata."""
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.utils.multiclass import type_of_target
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
        mean_absolute_error, mean_squared_error, r2_score, silhouette_score)
    from dataforge_sba.exceptions import AlgorithmError
    spec = next((s for s in SPECS if s.id == identifier), None)
    if spec is None: raise AlgorithmError(f"Unknown algorithm: {identifier}")
    try: seed = int(options.get("random_state", 42))
    except (TypeError, ValueError) as exc: raise AlgorithmError("Random seed must be an integer from 0 to 4294967295.") from exc
    if not 0 <= seed <= 4294967295: raise AlgorithmError("Random seed must be an integer from 0 to 4294967295.")
    if identifier == "kmeans":
        X = (frame[feature_columns].copy() if feature_columns is not None
             else frame.select_dtypes(include="number").copy())
        if not all(pd.api.types.is_numeric_dtype(X[column]) for column in X.columns):
            raise AlgorithmError("K-means requires numeric feature columns.")
        if X.empty or X.isna().all().all() or X.isna().all().any(): raise AlgorithmError("K-means needs usable values in every numeric feature column.")
        if np.isinf(X.to_numpy(dtype=float, na_value=np.nan)).any(): raise AlgorithmError("K-means features contain infinite values. Clean them before modeling.")
        k = int(options.get("n_clusters", 3))
        if not 2 <= k <= len(X): raise AlgorithmError("n_clusters must be between 2 and the number of rows.")
        scaler = options.get("scaler", "standard" if bool(options.get("scale", True)) else "none")
        if scaler not in {"standard", "minmax", "none"}: raise AlgorithmError("Choose standard, minmax, or none for K-means scaling.")
        model = _pipeline(KMeans(n_clusters=k, random_state=seed, n_init="auto"), X, scaler != "none", scaler, impute_strategies)
        labels = model.fit_predict(X)
        score = None
        transformed = model.named_steps["preprocessing"].transform(X)
        silhouette_sample = min(1000, len(labels))
        if 1 < len(set(labels)) < silhouette_sample:
            try:
                score = float(silhouette_score(transformed, labels, sample_size=silhouette_sample,
                                               random_state=seed))
            except ValueError:
                score = None
        return {"algorithm": identifier, "labels": labels.tolist(), "silhouette_score": score,
                "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict(), "model": model}
    if not target or target not in frame: raise AlgorithmError(f"{spec.name} requires a valid target column.")
    clean = frame.dropna(subset=[target]).copy(); y = clean.pop(target)
    if feature_columns is not None:
        if not isinstance(feature_columns, list) or not feature_columns:
            raise AlgorithmError("Choose at least one feature column.")
        if len(set(feature_columns)) != len(feature_columns):
            raise AlgorithmError("Feature columns must not be selected more than once.")
        missing = [column for column in feature_columns if column not in clean.columns]
        if missing: raise AlgorithmError(f"Unknown feature column(s): {', '.join(missing)}")
        X = clean[feature_columns].copy()
    else:
        X = clean
    if X.empty: raise AlgorithmError("Choose at least one feature column.")
    numeric_X = X.select_dtypes(include="number")
    if not numeric_X.empty and np.isinf(numeric_X.to_numpy(dtype=float, na_value=np.nan)).any():
        raise AlgorithmError("Numeric features contain infinite values. Convert or clean those values before modeling.")
    if identifier != "linear_regression" and type_of_target(y) not in ("binary", "multiclass"):
        raise AlgorithmError(f"{spec.name} needs a categorical target; '{target}' appears continuous. Choose a category column or use Linear Regression.")
    if identifier != "linear_regression" and y.value_counts().min() < 2:
        raise AlgorithmError("Each target category needs at least two rows for a reliable train/test split.")
    if len(X) < 5: raise AlgorithmError("At least five usable rows are required for a train/test split.")
    test_size = float(options.get("test_size", .2))
    if not 0.05 <= test_size <= .5: raise AlgorithmError("test_size must be between 0.05 and 0.5.")
    stratify = y if identifier != "linear_regression" and y.nunique() > 1 else None
    try: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
        random_state=seed, stratify=stratify)
    except ValueError as exc: raise AlgorithmError(str(exc)) from exc
    if identifier == "linear_regression":
        if not pd.api.types.is_numeric_dtype(y): raise AlgorithmError("Linear Regression target must be numeric.")
        estimator = LinearRegression(); default_scale = False
    elif identifier == "logistic_regression":
        max_iter = int(options.get("max_iter", 1000))
        if max_iter < 1: raise AlgorithmError("Maximum iterations must be at least 1.")
        estimator = LogisticRegression(max_iter=max_iter); default_scale = True
    elif identifier == "decision_tree":
        max_depth = options.get("max_depth", 4)
        if max_depth is not None and int(max_depth) < 1: raise AlgorithmError("Maximum tree depth must be at least 1.")
        estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=seed); default_scale = False
    else:
        n = int(options.get("n_neighbors", 5))
        if n < 1 or n > len(X_train): raise AlgorithmError("n_neighbors must fit within the training set.")
        estimator = KNeighborsClassifier(n_neighbors=n); default_scale = True
    scaler = options.get("scaler", "standard" if default_scale else "none")
    if scaler not in {"standard", "minmax", "none"}: raise AlgorithmError("Choose standard, minmax, or none for feature scaling.")
    model = _pipeline(estimator, X, scaler != "none", scaler, impute_strategies); model.fit(X_train, y_train); predictions = model.predict(X_test)
    if identifier == "linear_regression":
        mse = mean_squared_error(y_test, predictions)
        r2 = float(r2_score(y_test, predictions)) if len(y_test) >= 2 else None
        metrics = {"mae": float(mean_absolute_error(y_test, predictions)), "mse": float(mse), "rmse": float(np.sqrt(mse)), "r2": r2}
    else:
        metrics = {"accuracy": float(accuracy_score(y_test, predictions)),
                   "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                   "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                   "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0))}
    return {"algorithm": identifier, "metrics": metrics, "predictions": np.asarray(predictions).tolist(),
            "actual": np.asarray(y_test).tolist(), "model": model}
