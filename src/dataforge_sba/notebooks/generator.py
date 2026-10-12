"""Component-based notebook generation from trusted templates."""
from pathlib import Path
import nbformat


class NotebookGenerator:
    """Build a portable analysis notebook from a Dataset or CSV path."""
    def __init__(self, dataset, dataset_name: str | None = None):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.cleaning = []
        self.visualizations = []
        self.algorithms = []
        self.playground = []

    def add_cleaning(self, *, drop_duplicates: bool = False, operations: list[dict] | None = None):
        """Add reproducible cleaning operations while keeping the source file untouched."""
        self.cleaning.append({"drop_duplicates": drop_duplicates, "operations": list(operations or [])}); return self

    def add_visualizations(self, columns: list[str] | None = None, charts: list[str] | None = None):
        self.visualizations.append({"columns": list(columns or []),
                                    "charts": list(["histogram", "correlation"] if charts is None else charts)})
        return self

    def add_algorithm(self, identifier: str, **parameters):
        from dataforge_sba.algorithms import list_algorithms
        if identifier not in {item["id"] for item in list_algorithms()}: raise ValueError(f"Unknown algorithm: {identifier}")
        target = parameters.pop("target", None)
        self.algorithms.append((identifier, parameters, target)); return self

    def add_playground(self, settings: dict):
        """Include a reproducible version of the selected Playground experiment."""
        self.playground.append(dict(settings)); return self

    @staticmethod
    def _playground_code(settings: dict) -> str:
        algorithm = settings.get("algorithm", "kmeans")
        source = settings.get("source", "sample")
        seed = int(settings.get("random_state", 42))
        configured_imputation = settings.get("impute_strategies") or {}
        if algorithm not in {"kmeans", "linear_regression", "logistic_regression", "decision_tree", "knn"}:
            raise ValueError("Unknown Playground algorithm.")
        if source not in {"sample", "uploaded"}:
            raise ValueError("Unknown Playground data source.")
        if source == "sample":
            n = int(settings.get("n_samples", 300)); shape = settings.get("shape", "blobs")
            if algorithm == "linear_regression":
                data_setup = f"from sklearn.datasets import make_regression\n_X, _y = make_regression(n_samples={n}, n_features=1, noise={float(settings.get('noise', 15.0))!r}, random_state={seed})\n_play_df = pd.DataFrame({{'feature_1': _X[:, 0], 'target': _y}})"
            else:
                generator = ("make_classification" if shape == "blobs" and algorithm != "kmeans" else
                             {"blobs": "make_blobs", "moons": "make_moons", "circles": "make_circles"}.get(shape))
                if not generator: raise ValueError("Choose blobs, moons, or circles as the sample pattern.")
                args = f"n_samples={n}, random_state={seed}"
                if generator == "make_classification":
                    args += ", n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1, class_sep=1.2"
                elif shape == "blobs":
                    if algorithm == "kmeans": args += f", centers={int(settings.get('n_clusters', 3))}"
                    else: args += ", centers=3"
                else: args += ", noise=0.08" + (", factor=0.5" if shape == "circles" else "")
                y_column = "" if algorithm == "kmeans" else "\n_play_df['target'] = [f'class_{int(value)}' for value in _y]"
                data_setup = f"from sklearn.datasets import {generator}\n_X, _y = {generator}({args})\n_play_df = pd.DataFrame(_X, columns=['feature_1', 'feature_2']){y_column}"
            x_col, y_col = "feature_1", "feature_2" if algorithm != "linear_regression" else "target"
            target = "target"
        else:
            x_col = settings.get("feature_x"); y_col = settings.get("feature_y"); target = settings.get("target")
            if not x_col or (algorithm != "linear_regression" and algorithm != "kmeans" and not y_col):
                raise ValueError("Choose numeric Playground feature columns before generating the notebook.")
            data_setup = "_play_df = _model_data.copy() if " + repr(algorithm != "kmeans") + " and '_model_data' in globals() else df.copy()"
        x_repr, y_repr, target_repr = repr(x_col), repr(y_col), repr(target)
        if algorithm == "kmeans":
            y_axis = repr(y_col)
            features = f"[{x_repr}, {y_axis}]"
            k = int(settings.get("n_clusters", 3)); scale = bool(settings.get("scale", True))
            body = f"""from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
_features = {features}
_plot_df = _play_df[_features].dropna()
_X = _plot_df.to_numpy()
_scaler = StandardScaler().fit(_X) if {scale!r} else None
_X_fit = _scaler.transform(_X) if _scaler is not None else _X
_model = KMeans(n_clusters={k}, random_state={seed}, n_init='auto').fit(_X_fit)
_labels = _model.labels_
print('Cluster sizes:', pd.Series(_labels).value_counts().sort_index().to_dict())
if 1 < len(set(_labels)) < len(_labels):
    print('Silhouette:', silhouette_score(_X_fit, _labels, sample_size=min(1000, len(_labels)), random_state={seed}))
else:
    print('Silhouette is undefined for this clustering.')
plt.figure(figsize=(7, 5))
plt.scatter(_plot_df.iloc[:, 0], _plot_df.iloc[:, 1], c=_labels, cmap='viridis', alpha=0.75)
_centers = _scaler.inverse_transform(_model.cluster_centers_) if _scaler is not None else _model.cluster_centers_
plt.scatter(_centers[:, 0], _centers[:, 1], marker='X', s=150, c='black', label='Centroids'); plt.legend()
plt.xlabel({x_repr}); plt.ylabel({y_axis}); plt.title('K-means Playground experiment'); plt.show()
_elbow_values = []
for _candidate_k in range(1, min(8, len(_X_fit)) + 1):
    _candidate_model = KMeans(n_clusters=_candidate_k, random_state={seed}, n_init='auto').fit(_X_fit)
    _elbow_values.append(_candidate_model.inertia_)
plt.figure(figsize=(6, 4)); plt.plot(range(1, len(_elbow_values) + 1), _elbow_values, marker='o')
plt.xlabel('K (number of clusters)'); plt.ylabel('Within-cluster variation'); plt.title('Elbow curve'); plt.show()"""
        elif algorithm == "linear_regression":
            regression_imputer = "most_frequent" if configured_imputation.get(x_col) == "mode" else configured_imputation.get(x_col, "median")
            body = f"""from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
_X, _y = _play_df[[{x_repr}]], _play_df[{target_repr}]
_valid = _y.notna()
_X, _y = _X.loc[_valid], _y.loc[_valid]
_X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2, random_state={seed})
_model = make_pipeline(SimpleImputer(strategy={regression_imputer!r}), LinearRegression()).fit(_X_train, _y_train)
_pred = _model.predict(_X_test)
_r2 = r2_score(_y_test, _pred) if len(_y_test) >= 2 else None
print('MAE:', mean_absolute_error(_y_test, _pred)); print('RMSE:', mean_squared_error(_y_test, _pred) ** 0.5); print('R²:', _r2 if _r2 is not None else 'unavailable (need at least two test rows)')
_order = _X_test.iloc[:, 0].argsort()
plt.figure(figsize=(7, 5)); plt.scatter(_X_test.iloc[:, 0], _y_test, alpha=0.65); plt.plot(_X_test.iloc[_order, 0], _pred[_order], color='darkorange')
plt.xlabel({x_repr}); plt.ylabel({target_repr}); plt.title('Linear Regression fitted line (test rows)'); plt.show()
_low, _high = min(_y_test.min(), _pred.min()), max(_y_test.max(), _pred.max())
plt.figure(figsize=(5, 5)); plt.scatter(_y_test, _pred, alpha=0.7); plt.plot([_low, _high], [_low, _high], linestyle='--', color='darkorange')
plt.xlabel('Actual value'); plt.ylabel('Predicted value'); plt.title('Actual vs predicted'); plt.show()"""
        else:
            model_expr = {"logistic_regression": "LogisticRegression(max_iter=1000)", "decision_tree": f"DecisionTreeClassifier(max_depth={int(settings.get('max_depth', 4))}, random_state={seed})", "knn": f"KNeighborsClassifier(n_neighbors={int(settings.get('n_neighbors', 5))})"}[algorithm]
            feature_imputation = {column: ("most_frequent" if configured_imputation.get(column) == "mode" else configured_imputation.get(column, "median")) for column in (x_col, y_col)}
            imputer_steps = ", ".join(f"('impute_{index}', SimpleImputer(strategy={strategy!r}), [{index}])" for index, strategy in enumerate(feature_imputation.values()))
            classifier_pipeline = f"make_pipeline(ColumnTransformer([{imputer_steps}]), {('StandardScaler(), ' if algorithm in {'logistic_regression', 'knn'} else '')}{model_expr})"
            body = f"""from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
_features = [{x_repr}, {y_repr}]
_plot_df = _play_df[_features + [{target_repr}]].dropna(subset=[{target_repr}])
_X, _y = _plot_df[_features], _plot_df[{target_repr}]
_stratify = _y if _y.nunique() > 1 and _y.value_counts().min() >= 2 else None
_X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2, random_state={seed}, stratify=_stratify)
_model = {classifier_pipeline}.fit(_X_train, _y_train)
_pred = _model.predict(_X_test)
print('Accuracy:', accuracy_score(_y_test, _pred)); print(classification_report(_y_test, _pred, zero_division=0))
_x_grid = np.linspace(_plot_df.iloc[:, 0].min(), _plot_df.iloc[:, 0].max(), 35)
_y_grid = np.linspace(_plot_df.iloc[:, 1].min(), _plot_df.iloc[:, 1].max(), 35)
_gx, _gy = np.meshgrid(_x_grid, _y_grid)
_grid = pd.DataFrame({{_features[0]: _gx.ravel(), _features[1]: _gy.ravel()}})
_region = pd.factorize(_model.predict(_grid))[0].reshape(_gx.shape)
plt.figure(figsize=(7, 5)); plt.contourf(_gx, _gy, _region, levels=len(_model.classes_), cmap='viridis', alpha=0.18)
plt.scatter(_plot_df.iloc[:, 0], _plot_df.iloc[:, 1], c=pd.factorize(_plot_df[{target_repr}])[0], cmap='viridis', alpha=0.75)
plt.xlabel({x_repr}); plt.ylabel({y_repr}); plt.title('{algorithm.replace('_', ' ').title()} Playground experiment'); plt.show()"""
        return data_setup + "\n\n" + body

    def build(self):
        source = self.dataset_name or getattr(self.dataset, "source", None) or "your_dataset.csv"
        portable_name = Path(source).name
        loader = "pd.read_excel(DATA_PATH)" if portable_name.lower().endswith(".xlsx") else "pd.read_csv(DATA_PATH)"
        cells = [nbformat.v4.new_markdown_cell("# DataForge analysis\n\nA reproducible, editable starting point for exploring your dataset."),
                 nbformat.v4.new_markdown_cell("## 1. Load and inspect data\n\nUpdate `DATA_PATH` to the location of your CSV or Excel file."),
                 nbformat.v4.new_code_cell(f"from pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nDATA_PATH = {portable_name!r}\ndf = {loader}\ndisplay(df.head())\nprint('Shape:', df.shape)\ndisplay(df.dtypes.to_frame('type'))\ndisplay(df.isna().sum().to_frame('missing'))\nprint('Duplicate rows:', int(df.duplicated().sum()))\nprint('Constant columns:', [c for c in df.columns if df[c].nunique(dropna=False) <= 1])\n_numeric_summary = df.select_dtypes(include='number')\nif not _numeric_summary.empty:\n    display(_numeric_summary.describe().T)\n    display(_numeric_summary.corr().fillna(0))\n_categorical_summary = df.select_dtypes(exclude='number')\nfor _column in _categorical_summary.columns:\n    display(_categorical_summary[_column].value_counts(dropna=False).head(10).rename('count').to_frame())"),
                 nbformat.v4.new_markdown_cell("## 2. Descriptive statistics"),
                 nbformat.v4.new_code_cell("display(df.describe(include='all').T)\nprint('Duplicate rows:', int(df.duplicated().sum()))")]
        cleaning_code = []
        has_supervised_models = (any(identifier != "kmeans" for identifier, _, _ in self.algorithms)
                                 or any(settings.get("algorithm") != "kmeans" for settings in self.playground))
        for config in self.cleaning:
            operations = list(config.get("operations", []))
            if config.get("drop_duplicates") and not any(op.get("type") == "drop_duplicates" for op in operations):
                operations.insert(0, {"type": "drop_duplicates"})
            for op in operations:
                kind = op.get("type")
                columns = op.get("columns") or []
                if kind == "drop_duplicates":
                    cleaning_code.append("df = df.drop_duplicates().reset_index(drop=True)")
                elif kind == "impute":
                    if has_supervised_models:
                        cleaning_code.append("print('Selected imputation is fitted on training rows inside supervised model pipelines; the exploratory DataFrame keeps its missing values.')")
                        continue
                    for column in columns:
                        strategy = op.get("strategies", {}).get(column)
                        if strategy in {"mean", "median"}:
                            cleaning_code.append(f"df[{column!r}] = df[{column!r}].fillna(df[{column!r}].{strategy}())")
                        elif strategy == "mode":
                            cleaning_code.append(f"_mode = df[{column!r}].mode(dropna=True)\ndf[{column!r}] = df[{column!r}].fillna(_mode.iloc[0] if not _mode.empty else pd.NA)")
                elif kind == "drop_missing_rows":
                    cleaning_code.append(f"df = df.dropna(subset={columns!r}).reset_index(drop=True)")
                elif kind == "drop_missing_columns":
                    cleaning_code.append(f"df = df.drop(columns=[c for c in {columns!r} if df[c].isna().any()])")
                elif kind == "convert":
                    column, target = op.get("column"), op.get("target")
                    if target == "numeric":
                        cleaning_code.append(f"_before_valid = df[{column!r}].notna().sum()\ndf[{column!r}] = pd.to_numeric(df[{column!r}], errors='coerce')\nprint('Values that could not be converted:', int(_before_valid - df[{column!r}].notna().sum()))")
                    elif target == "datetime":
                        cleaning_code.append(f"_before_valid = df[{column!r}].notna().sum()\ndf[{column!r}] = pd.to_datetime(df[{column!r}], errors='coerce')\nprint('Values that could not be converted:', int(_before_valid - df[{column!r}].notna().sum()))")
                    elif target == "category": cleaning_code.append(f"df[{column!r}] = df[{column!r}].astype('category')")
                    elif target == "boolean":
                        cleaning_code.append(f"_before_valid = df[{column!r}].notna().sum()\n_boolean_map = {{'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 'y': True, 'n': False}}\ndf[{column!r}] = df[{column!r}].astype('string').str.strip().str.lower().map(_boolean_map).astype('boolean')\nprint('Values that could not be converted:', int(_before_valid - df[{column!r}].notna().sum()))")
                elif kind == "encode":
                    if has_supervised_models:
                        cleaning_code.append("print('Categorical encoding for supervised experiments is handled inside each model pipeline after the training split to prevent data leakage.')")
                    elif op.get("method") == "ordinal":
                        for column in columns:
                            order = [str(value) for value in (op.get("category_order") or [])]
                            cleaning_code.append(f"df[{column!r}] = df[{column!r}].astype('string').map({{value: index for index, value in enumerate({order!r})}}).astype('Int64')")
                    else:
                        cleaning_code.append(f"df = pd.get_dummies(df, columns={columns!r}, dtype=int)")
                elif kind == "outliers":
                    for column in columns:
                        cleaning_code.append(f"_q1, _q3 = df[{column!r}].quantile([0.25, 0.75])\n_iqr = _q3 - _q1\nprint({column!r}, 'potential outliers:', int(((df[{column!r}] < _q1 - 1.5 * _iqr) | (df[{column!r}] > _q3 + 1.5 * _iqr)).sum()))")
        if cleaning_code:
            cells.extend([nbformat.v4.new_markdown_cell("## 3. Selected cleaning\n\nThe original file is unchanged. These steps modify only the notebook's in-memory DataFrame."),
                          nbformat.v4.new_code_cell("\n".join(cleaning_code) + "\nprint('Shape after cleaning:', df.shape)\nprint('Missing cells after cleaning:', int(df.isna().sum().sum()))")])
        if has_supervised_models:
            cells.append(nbformat.v4.new_code_cell("_model_data = df.copy()"))
        for viz in self.visualizations:
            columns = viz["columns"]
            charts = [name for name in viz["charts"] if name in {"histogram", "boxplot", "scatter", "correlation", "bar", "frequency"}]
            sections = ["## Selected exploratory visualizations"]
            code = [f"_viz_columns = {columns!r}",
                    "_numeric = df[_viz_columns].select_dtypes(include='number') if _viz_columns else df.select_dtypes(include='number')",
                    "_categorical = df[_viz_columns].select_dtypes(exclude='number') if _viz_columns else df.select_dtypes(exclude='number')"]
            if "histogram" in charts:
                code.extend(["if not _numeric.empty:", "    _numeric.hist(figsize=(12, 8), bins=20)", "    plt.suptitle('Numeric feature distributions')", "    plt.tight_layout()", "    plt.show()"])
            if "boxplot" in charts:
                code.extend(["if not _numeric.empty:", "    _numeric.plot(kind='box', subplots=True, layout=(1, len(_numeric.columns)), figsize=(max(6, 3 * len(_numeric.columns)), 4), sharey=False)", "    plt.suptitle('Numeric boxplots')", "    plt.tight_layout()", "    plt.show()"])
            if "scatter" in charts:
                code.extend(["if len(_numeric.columns) >= 2:", "    _numeric.plot(kind='scatter', x=_numeric.columns[0], y=_numeric.columns[1], figsize=(6, 5), title='Selected numeric features')", "    plt.show()", "else:", "    print('Scatterplot skipped: choose at least two numeric columns.')"])
            if "correlation" in charts:
                code.extend(["if _numeric.shape[1] >= 2:", "    _corr = _numeric.corr()", "    plt.figure(figsize=(max(6, len(_corr.columns)), max(4, len(_corr.columns) * .7)))", "    plt.imshow(_corr, vmin=-1, vmax=1, cmap='coolwarm')", "    plt.colorbar(label='Correlation')", "    plt.xticks(range(len(_corr.columns)), _corr.columns, rotation=45, ha='right')", "    plt.yticks(range(len(_corr.columns)), _corr.columns)", "    plt.title('Numeric correlation heatmap')", "    plt.tight_layout()", "    plt.show()", "else:", "    print('Correlation heatmap skipped: choose at least two numeric columns.')"])
            if "bar" in charts or "frequency" in charts:
                code.extend(["for _column in _categorical.columns:", "    _all_counts = _categorical[_column].astype('string').fillna('(missing)').value_counts()", "    _counts = _all_counts.head(20).copy()", "    if len(_all_counts) > 20:", "        _counts.loc['Other categories'] = int(_all_counts.iloc[20:].sum())", "    if not _counts.empty:", "        display(_counts.to_frame('count'))"])
                if "bar" in charts:
                    code.extend(["        _counts.plot(kind='bar', figsize=(8, 4), title=f'Categories in {_column}')", "        plt.xlabel(_column)", "        plt.ylabel('Count')", "        plt.tight_layout()", "        plt.show()"])
            cells.extend([nbformat.v4.new_markdown_cell(sections[0]), nbformat.v4.new_code_cell("\n".join(code))])
        for settings in self.playground:
            cells.extend([nbformat.v4.new_markdown_cell("## Machine Learning Playground experiment\n\nThis cell recreates the selected experiment and displays its evaluation and chart."),
                          nbformat.v4.new_code_cell(self._playground_code(settings))])
        supervised_ids = [identifier for identifier, _, _ in self.algorithms if identifier != "kmeans"]
        if supervised_ids:
            cells.append(nbformat.v4.new_code_cell("model_comparison = []"))
        for identifier, params, selected_target in self.algorithms:
            cells.append(nbformat.v4.new_markdown_cell(f"## {identifier.replace('_', ' ').title()}\n\nThis experiment uses a scikit-learn pipeline. Configure feature and target columns before running."))
            if identifier == "kmeans":
                k = int(params.get("n_clusters", 3))
                seed = int(params.get("random_state", 42)); features = params.get("feature_columns")
                feature_expr = f"df[{features!r}]" if features else "df.select_dtypes(include='number')"
                scale = params.get("scaler", "standard")
                scale_expr = "StandardScaler().fit_transform(X)" if scale == "standard" else "MinMaxScaler().fit_transform(X)" if scale == "minmax" else "X"
                cells.append(nbformat.v4.new_code_cell(f"from sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import silhouette_score\n\nX = {feature_expr}\nX = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=X.columns, index=X.index)\nX_plot = X.copy()\nX_scaled = {scale_expr}\nmodel = KMeans(n_clusters={k}, random_state={seed}, n_init='auto')\nlabels = model.fit_predict(X_scaled)\nprint('Cluster sizes:', pd.Series(labels).value_counts().sort_index().to_dict())\n_silhouette_sample = min(1000, len(labels))\nif 1 < len(set(labels)) < _silhouette_sample:\n    try:\n        print('Silhouette:', silhouette_score(X_scaled, labels, sample_size=_silhouette_sample, random_state={seed}))\n    except ValueError:\n        print('Silhouette is unavailable for this sample.')\nelse:\n    print('Silhouette is undefined for this cluster assignment.')\nif X_plot.shape[1] >= 2:\n    centers = model.cluster_centers_\n    if {scale!r} == 'standard': centers = StandardScaler().fit(X_plot).inverse_transform(centers)\n    elif {scale!r} == 'minmax': centers = MinMaxScaler().fit(X_plot).inverse_transform(centers)\n    X_plot.iloc[:, :2].plot(kind='scatter', x=X_plot.columns[0], y=X_plot.columns[1], c=labels, colormap='viridis', figsize=(6, 5), title='K-means clusters (model used selected numeric features)')\n    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=150, c='black', label='Centroids')\n    plt.legend(); plt.show()\nelse:\n    print('No cluster chart: choose at least two numeric features to display a 2D chart.')"))
            else:
                model_expression = {
                    "linear_regression": "LinearRegression()",
                    "logistic_regression": f"LogisticRegression(max_iter={int(params.get('max_iter', 1000))})",
                    "decision_tree": f"DecisionTreeClassifier(max_depth={int(params.get('max_depth', 4))}, random_state={int(params.get('random_state', 42))})",
                    "knn": f"KNeighborsClassifier(n_neighbors={int(params.get('n_neighbors', 5))})",
                }[identifier]
                features = params.get("feature_columns")
                features_expr = f"{features!r}" if features else "None"
                test_size = float(params.get("test_size", .2)); seed = int(params.get("random_state", 42))
                scaler = params.get("scaler", "none" if identifier in {"linear_regression", "decision_tree"} else "standard")
                strategies = params.get("impute_strategies") or {}
                scaler_step = ", StandardScaler()" if scaler == "standard" else ", MinMaxScaler()" if scaler == "minmax" else ""
                numeric_pipeline = "[(f'num_{_strategy}', make_pipeline(SimpleImputer(strategy=_strategy)" + scaler_step + "), [c for c in numeric if ('most_frequent' if _impute_strategies.get(c) == 'mode' else _impute_strategies.get(c, 'median')) == _strategy]) for _strategy in ('median', 'mean', 'most_frequent') if any(('most_frequent' if _impute_strategies.get(c) == 'mode' else _impute_strategies.get(c, 'median')) == _strategy for c in numeric)]"
                metrics = ("print('MAE:', mean_absolute_error(y_test, predictions))\n"
                           "print('RMSE:', mean_squared_error(y_test, predictions) ** 0.5)\n"
                           "_r2 = r2_score(y_test, predictions) if len(y_test) >= 2 else None\n"
                           "print('R²:', _r2 if _r2 is not None else 'unavailable (need at least two test rows)')\n"
                           f"model_comparison.append({{'algorithm': {identifier!r}, 'MAE': mean_absolute_error(y_test, predictions), 'RMSE': mean_squared_error(y_test, predictions) ** 0.5, 'R²': _r2}})") if identifier == "linear_regression" else (
                           "print('Accuracy:', accuracy_score(y_test, predictions))\n"
                           "print(classification_report(y_test, predictions, zero_division=0))\n"
                           f"model_comparison.append({{'algorithm': {identifier!r}, 'accuracy': accuracy_score(y_test, predictions), 'precision_weighted': precision_score(y_test, predictions, average='weighted', zero_division=0), 'recall_weighted': recall_score(y_test, predictions, average='weighted', zero_division=0), 'f1_weighted': f1_score(y_test, predictions, average='weighted', zero_division=0)}})")
                experiment = f"""    X, y = _model_data.drop(columns=[TARGET]), _model_data[TARGET]
    FEATURES = {features_expr}
    if FEATURES is not None:
        X = X[FEATURES]
    _valid = y.notna()
    X, y = X.loc[_valid], y.loc[_valid]
    _stratify = y if {identifier != 'linear_regression'!r} and y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size!r}, random_state={seed}, stratify=_stratify)
    model = {model_expression}
    numeric = X.select_dtypes(include='number').columns
    categorical = X.columns.difference(numeric)
    _impute_strategies = {strategies!r}
    prep = ColumnTransformer({numeric_pipeline} + ([('cat', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')), categorical)] if len(categorical) else []))
    pipeline = make_pipeline(prep, model).fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    {metrics.replace(chr(10), chr(10) + '    ')}"""
                if identifier == "linear_regression":
                    target_check = (
                        "    if not pd.api.types.is_numeric_dtype(y):\n"
                        "        print('Skipped: Linear Regression needs a numeric target. Choose a numeric column, or use a classification algorithm for labels such as species.')\n"
                        "    else:"
                    )
                else:
                    target_check = (
                        "    from sklearn.utils.multiclass import type_of_target\n"
                        "    if type_of_target(y) not in ('binary', 'multiclass'):\n"
                        f"        print('Skipped: {identifier.replace('_', ' ').title()} needs a categorical target, such as species. The selected target looks continuous; use Linear Regression for numeric values such as tip.')\n"
                        "    else:"
                    )
                experiment = experiment.replace(
                    "    X, y = _model_data.drop(columns=[TARGET]), _model_data[TARGET]",
                    "    X, y = _model_data.drop(columns=[TARGET]), _model_data[TARGET]\n" + target_check,
                )
                lines = experiment.splitlines()
                start = next(i for i, line in enumerate(lines) if line.strip() == "else:")
                experiment = "\n".join(lines[:start + 1] + ["    " + line for line in lines[start + 1:]])
                code = f"""# Target selected in the DataForge Notebook Builder.
TARGET = {selected_target!r}
if TARGET is None:
    print('Choose TARGET before running the {identifier} experiment.')
else:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score
{experiment}"""
                cells.append(nbformat.v4.new_code_cell(code))
        if len(supervised_ids) > 1:
            cells.append(nbformat.v4.new_markdown_cell("## Model comparison\n\nCompare these held-out test-set metrics. Scores depend on this dataset and split; they do not guarantee future performance."))
            cells.append(nbformat.v4.new_code_cell("if model_comparison:\n    display(pd.DataFrame(model_comparison).set_index('algorithm'))\nelse:\n    print('No compatible supervised model completed.')"))
        cells.append(nbformat.v4.new_markdown_cell("## Next steps\n\nReview data quality, try justified feature choices, and compare multiple metrics before drawing conclusions."))
        return nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})

    def save(self, path: str | Path) -> Path:
        """Write a validated .ipynb file and return its path."""
        notebook = self.build(); nbformat.validate(notebook)
        target = Path(path); nbformat.write(notebook, target); return target
