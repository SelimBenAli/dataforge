# DataForge

DataForge is a local-first Python toolkit for inspecting datasets, learning core machine learning concepts, and generating editable Jupyter notebooks. Its Python library is the product; Flask is an optional local dashboard that calls the same library services.

## V1 features

- Load and inspect CSV and `.xlsx` datasets without changing the source file.
- Review schema, missing values, duplicates, descriptive statistics, constant columns, category frequencies, and numeric correlations.
- In the dashboard, preview and apply missing-value imputation or removal, duplicate removal, type conversion, categorical encoding, and IQR outlier detection. Transformations affect the session's in-memory dataset; the uploaded source file is not overwritten.
- Create selectable histograms, boxplots, scatterplots, correlation heatmaps, and category bar/frequency charts.
- Run Linear Regression, Logistic Regression, K-means, Decision Tree, and KNN using shared train/test and preprocessing pipelines.
- Generate structured, editable `.ipynb` notebooks from Python or the dashboard, including selected cleaning steps, EDA, features, targets, model parameters, and optionally the current Playground experiment.
- Explore actual K-means, regression, and classification experiments in the Playground, using reproducible built-in samples or a session's uploaded dataset. K-means includes cluster assignments, centroids, sizes, an elbow curve, and a silhouette score when valid. Linear Regression includes a fitted line, held-out actual-versus-predicted points, and evaluation metrics.
- Launch a local dashboard on an explicitly selected port.

The dashboard is local and has a 25 MB upload limit. Upload CSV or `.xlsx` files; use the **Remove uploaded data** button to delete the current session's uploaded file. A new upload replaces and removes that session's previous file. If the browser session is lost before removal, stop DataForge and delete files from `.dataforge/uploads` manually. Downloaded notebook copies are temporary and are removed after transfer.

Cleaning can be staged, previewed, applied, and reset to the uploaded original. Outlier detection only flags potential anomalies; it never removes them. Ordinal encoding requires the user to explicitly provide the real category order. Supervised-model preprocessing (including one-hot encoding and scaling) is fitted inside the training pipeline after the train/test split. The Notebook Builder tests the selected workflow before enabling download, and the generated notebook remains editable and must be run by the user in Jupyter.

The K-means Playground offers an elbow curve so learners can compare K values, plus silhouette where mathematically valid. Step-by-step centroid iteration animation is not included in V1. Playground chart features are limited to one numeric predictor for Linear Regression and two numeric features for the other algorithms; the visualized features are the features used for the Playground model.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dashboard,visualization]"
```

For library and dashboard development:

```bash
python -m pip install -e ".[dev,dashboard,visualization]"
python -m pytest
```

Excel `.xlsx` needs `openpyxl`, included in the `dashboard` extra. The library itself does not import Flask. Plotting in notebooks can use Matplotlib/Seaborn from the `visualization` extra.

## Launch the dashboard

```bash
dataforge-sba dashboard
dataforge-sba dashboard --port 8080
```

The default bind address is `127.0.0.1`; debug mode is off. If the selected port is unavailable, DataForge reports the issue instead of silently changing ports. Stop with Ctrl+C. The dashboard upload limit is 25 MB. Uploaded data stays in the local `.dataforge/uploads` folder and each browser session refers to its own upload.

## Python API

```python
from dataforge_sba import Dataset
from dataforge_sba.cleaning import impute_missing, missing_summary

dataset = Dataset("data.csv")
print(dataset.shape)
print(dataset.schema())
report = dataset.explore()

cleaned = impute_missing(dataset.frame, {"age": "median", "city": "mode"})
print(missing_summary(cleaned))
```

To fit a model, pass a DataFrame with the target column named explicitly:

```python
from dataforge_sba.algorithms import run_algorithm

result = run_algorithm("linear_regression", dataset.frame, target="sales")
print(result["metrics"])
```

The returned model is a fitted scikit-learn pipeline. Preprocessing is fitted on training data only. Classification metrics are weighted averages; inspect the task and dataset before interpreting scores.

## Generate a notebook

```python
from dataforge_sba import Dataset
from dataforge_sba.notebooks import NotebookGenerator

dataset = Dataset("data.csv")
generator = NotebookGenerator(dataset)
generator.add_cleaning(drop_duplicates=True)
generator.add_visualizations()
generator.add_algorithm("kmeans", n_clusters=4)
generator.save("analysis.ipynb")
```

The generated notebook uses a portable dataset filename and a clear `DATA_PATH` variable. Move the original dataset beside the notebook or update `DATA_PATH` in its first code cell. Supervised experiments include a target and optional selected features; K-means does not ask for a target. The dashboard also offers algorithm-specific parameters, split fraction, seed, scaler choice, selected EDA charts, cleaning steps, and optional Playground export. Dashboard downloads are preflighted against the uploaded data and structurally validated; the notebook is not fully executed until you run its cells in Jupyter.

## K-means Playground from Python

```python
from dataforge_sba.playground import run_kmeans

experiment = run_kmeans(shape="moons", n_clusters=2, random_state=7)
print(experiment["silhouette"], experiment["cluster_sizes"])
```

Cluster labels are arbitrary identifiers, not rankings. Scaling affects distances, and silhouette scores are unavailable when the mathematical requirements are not met.

## Algorithm catalog

| Algorithm ID | Task | Current implementation |
|---|---|---|
| `linear_regression` | Numeric prediction | Train/test split; MAE, MSE, RMSE, R² |
| `logistic_regression` | Classification | Binary/multiclass; accuracy, weighted precision/recall/F1 |
| `kmeans` | Clustering | Numeric features, optional scaling, cluster sizes, silhouette score |
| `decision_tree` | Classification | Configurable depth and reproducible split |
| `knn` | Classification | Configurable neighbor count and scaled features |

Use `dataforge_sba.algorithms.list_algorithms()` to retrieve registry metadata. Extend the catalog through the central algorithm registry; UI labels are populated from its metadata endpoint.

## Privacy

DataForge runs locally and sends no dataset data to external services. It does not execute uploaded Python code or overwrite the source dataset. The browser dashboard uses first-party local requests only. Keep `.dataforge/` private if your data is sensitive, and remove temporary uploads when finished.

## Roadmap

The specification's V1 roadmap includes stronger cleaning previews, expanded charts, tree/linear/KNN learning Playground experiences, deeper parameter controls, model comparison, and broader dashboard workflows. Future versions may add further algorithms, PCA, cross-validation, experiment persistence, and optional AI providers. Those items are not represented as completed features here.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). DataForge is released under the MIT License; see [LICENSE](LICENSE).
"# dataforge" 
