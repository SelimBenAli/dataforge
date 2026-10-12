"""Actual scikit-learn experiments for the Playground and Python API."""
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification, make_regression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from dataforge_sba.algorithms import run_algorithm


def run_kmeans(shape: str = "blobs", n_samples: int = 300, n_clusters: int = 3,
               random_state: int = 42, scale: bool = True) -> dict:
    """Generate a 2D sample and run real K-means, returning JSON-safe results."""
    if not 0 <= int(random_state) <= 4294967295: raise ValueError("Random seed must be from 0 to 4294967295.")
    generators = {"blobs": lambda: make_blobs(n_samples=n_samples, centers=max(n_clusters, 2), random_state=random_state),
                  "moons": lambda: make_moons(n_samples=n_samples, noise=.08, random_state=random_state),
                  "circles": lambda: make_circles(n_samples=n_samples, noise=.06, factor=.5, random_state=random_state)}
    if shape not in generators: raise ValueError("shape must be blobs, moons, or circles")
    X, _ = generators[shape](); data = StandardScaler().fit_transform(X) if scale else X
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(data)
    labels = model.labels_
    score = float(silhouette_score(data, labels)) if 1 < len(set(labels)) < len(data) else None
    return {"points": X.round(5).tolist(), "labels": labels.tolist(),
            "centroids": (model.cluster_centers_ if not scale else StandardScaler().fit(X).inverse_transform(model.cluster_centers_)).round(5).tolist(),
            "silhouette": score, "cluster_sizes": np.bincount(labels, minlength=n_clusters).tolist(),
            "explanation": "K-means assigns each point to its nearest centroid. Cluster IDs are arbitrary labels; scaling changes how distances are measured."}


def _sample_frame(algorithm: str, shape: str, n_samples: int, seed: int, noise: float,
                 n_clusters: int) -> tuple[pd.DataFrame, str, list[str]]:
    """Create a small, deterministic example table for the selected experiment."""
    if algorithm == "linear_regression":
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=seed)
        return pd.DataFrame({"feature_1": X[:, 0], "target": y}), "target", ["feature_1"]
    if algorithm == "kmeans":
        generators = {
            "blobs": lambda: make_blobs(n_samples=n_samples, centers=n_clusters, random_state=seed),
            "moons": lambda: make_moons(n_samples=n_samples, noise=.08, random_state=seed),
            "circles": lambda: make_circles(n_samples=n_samples, noise=.06, factor=.5, random_state=seed),
        }
        if shape not in generators: raise ValueError("Choose blobs, moons, or circles for K-means.")
        X, _ = generators[shape]()
        return pd.DataFrame(X, columns=["feature_1", "feature_2"]), "", ["feature_1", "feature_2"]
    if shape == "moons":
        X, y = make_moons(n_samples=n_samples, noise=.12, random_state=seed)
    elif shape == "circles":
        X, y = make_circles(n_samples=n_samples, noise=.08, factor=.5, random_state=seed)
    elif shape == "blobs":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
            n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
            class_sep=1.2, random_state=seed)
    else:
        raise ValueError("Choose blobs, moons, or circles for classification.")
    frame = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    frame["target"] = [f"class_{int(value)}" for value in y]
    return frame, "target", ["feature_1", "feature_2"]


def _elbow_curve(frame: pd.DataFrame, features: list[str], seed: int, scale: bool) -> tuple[list[dict], bool]:
    """Calculate real K-means inertia values across candidate K values."""
    data = frame[features].dropna().to_numpy(dtype=float)
    sampled = len(data) > 2000
    if sampled:
        data = data[np.random.default_rng(seed).choice(len(data), 2000, replace=False)]
    if scale: data = StandardScaler().fit_transform(data)
    scores = []
    for k in range(1, min(8, len(data)) + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit(data)
        scores.append({"k": k, "inertia": float(model.inertia_)})
    return scores, sampled


def run_experiment(algorithm: str, *, source: str = "sample", frame: pd.DataFrame | None = None,
                   shape: str = "blobs", n_samples: int = 300, random_state: int = 42,
                   noise: float = 15.0, n_clusters: int = 3, scale: bool = True,
                   max_depth: int = 4, n_neighbors: int = 5, target: str | None = None,
                   feature_x: str | None = None, feature_y: str | None = None,
                   impute_strategies: dict | None = None) -> dict:
    """Run a real selected algorithm on sample data or a supplied DataFrame.

    Returns JSON-safe points, evaluation, and the visualization data for the UI.
    """
    if source not in {"sample", "uploaded"}:
        raise ValueError("source must be sample or uploaded")
    if algorithm not in {"kmeans", "linear_regression", "logistic_regression", "decision_tree", "knn"}:
        raise ValueError("Choose one of the available Playground algorithms.")
    if not 0 <= int(random_state) <= 4294967295:
        raise ValueError("Random seed must be from 0 to 4294967295.")
    if source == "sample" and not 50 <= int(n_samples) <= 1000:
        raise ValueError("Choose between 50 and 1000 sample rows.")
    if not 2 <= int(n_clusters) <= 12:
        raise ValueError("Choose between 2 and 12 clusters.")
    if not 1 <= int(max_depth) <= 20:
        raise ValueError("Tree depth must be between 1 and 20.")
    if not 1 <= int(n_neighbors) <= 50:
        raise ValueError("Choose between 1 and 50 neighbors.")
    if source == "sample":
        frame, target, features = _sample_frame(algorithm, shape, n_samples,
            random_state, noise, n_clusters)
        feature_x = features[0]
        feature_y = features[1] if len(features) > 1 else None
    else:
        if frame is None: raise ValueError("Upload a dataset before choosing uploaded CSV data.")
        frame = frame.copy(deep=True).reset_index(drop=True)
        if not feature_x: raise ValueError("Choose the first feature column.")
        if algorithm == "linear_regression": features = [feature_x]
        else:
            if not feature_y: raise ValueError("Choose the second numeric feature column.")
            if feature_x == feature_y: raise ValueError("Choose two different feature columns.")
            features = [feature_x, feature_y]
    if frame is None: raise ValueError("No dataset is available for this experiment.")
    if algorithm != "kmeans":
        if not target: raise ValueError("Choose a target column for this supervised experiment.")
        if target in features: raise ValueError("The target column cannot also be a feature.")
    for feature in features:
        if feature not in frame.columns: raise ValueError(f"Unknown feature column: {feature}")
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            raise ValueError(f"'{feature}' is not numeric. Choose numeric columns for the chart features.")

    if algorithm == "kmeans":
        result = run_algorithm("kmeans", frame, feature_columns=features,
                               n_clusters=n_clusters, scale=scale, random_state=random_state,
                               impute_strategies=impute_strategies)
        valid = frame[features].notna().all(axis=1)
        labels = result["labels"]
        points = [{"x": float(frame.at[i, features[0]]), "y": float(frame.at[i, features[1]]),
                   "label": str(labels[i])} for i in frame.index[valid][:1000]]
        model = result["model"]
        centers = model.named_steps["model"].cluster_centers_
        if scale:
            transformers = model.named_steps["preprocessing"].named_transformers_
            numeric_pipeline = next(value for name, value in transformers.items() if name.startswith("numeric_"))
            scaler = numeric_pipeline.named_steps.get("scaler")
            if scaler is not None: centers = scaler.inverse_transform(centers)
        elbow, elbow_sampled = _elbow_curve(frame, features, random_state, scale)
        return {"algorithm": algorithm, "points": points,
                "axis_x": features[0], "axis_y": features[1],
                "centroids": centers[:, :2].round(5).tolist(),
                "elbow": elbow, "elbow_sampled": elbow_sampled,
                "classes": sorted({point["label"] for point in points}),
                "cluster_sizes": result["cluster_sizes"],
                "metrics": {"silhouette": result["silhouette_score"],
                            "cluster_count": len(result["cluster_sizes"]),
                            "rows_shown": len(points), "rows_used": len(frame)},
                "explanation": "K is the number of groups requested. Each centroid is a group's center. The elbow curve shows within-group variation as K changes; a bend can suggest a useful range, but there is no guaranteed best K. Silhouette compares within-cluster cohesion with separation (closer to 1 is usually better). Cluster IDs are arbitrary labels, not real-world categories. Scaling changes distance calculations."}

    if algorithm == "linear_regression":
        result = run_algorithm(algorithm, frame, target=target, feature_columns=features,
                               random_state=random_state, impute_strategies=impute_strategies)
        observed = frame.dropna(subset=features + [target])
        shown = observed.head(1000)
        low, high = float(observed[feature_x].min()), float(observed[feature_x].max())
        grid_x = np.linspace(low, high, 80)
        grid = pd.DataFrame({feature_x: grid_x})
        curve_y = result["model"].predict(grid)
        actual_vs_predicted = [{"actual": float(actual), "predicted": float(predicted)}
                               for actual, predicted in zip(result["actual"], result["predictions"])]
        return {"algorithm": algorithm,
                "points": [{"x": float(row[feature_x]), "y": float(row[target])} for _, row in shown.iterrows()],
                "axis_x": feature_x, "axis_y": target,
                "curve": [{"x": float(x), "y": float(y)} for x, y in zip(grid_x, curve_y)],
                "actual_vs_predicted": actual_vs_predicted,
                "metrics": result["metrics"], "rows_shown": len(shown),
                "explanation": "The orange line is fitted using training rows; scores and actual-vs-predicted dots use held-out test rows. MAE and RMSE summarize typical prediction error (lower is better); R² compares against predicting the average and can be misleading on tiny or unusual samples. A fitted line shows association, not causation."}

    result = run_algorithm(algorithm, frame, target=target, feature_columns=features,
        random_state=random_state, max_depth=max_depth, n_neighbors=n_neighbors,
        impute_strategies=impute_strategies)
    observed = frame.dropna(subset=features + [target])
    shown = observed.head(1000)
    classes = sorted({str(value) for value in observed[target].unique()})
    low_x, high_x = float(observed[features[0]].min()), float(observed[features[0]].max())
    low_y, high_y = float(observed[features[1]].min()), float(observed[features[1]].max())
    xs, ys = np.linspace(low_x, high_x, 35), np.linspace(low_y, high_y, 35)
    gx, gy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({features[0]: gx.ravel(), features[1]: gy.ravel()})
    predicted = result["model"].predict(grid)
    boundary = [{"x": float(x), "y": float(y), "label": str(label)}
                for x, y, label in zip(gx.ravel(), gy.ravel(), predicted)]
    points = [{"x": float(row[features[0]]), "y": float(row[features[1]]),
               "label": str(row[target])} for _, row in shown.iterrows()]
    return {"algorithm": algorithm, "points": points, "boundary": boundary,
            "axis_x": features[0], "axis_y": features[1],
            "classes": classes, "metrics": result["metrics"], "rows_shown": len(points),
            "explanation": "Background colors show model predictions; point colors show actual labels. Scores are measured on held-out test rows. High scores on this dataset do not guarantee future performance."}
