import nbformat
import pandas as pd
import pytest

from dataforge_sba import Dataset
from dataforge_sba.algorithms import list_algorithms, run_algorithm
from dataforge_sba.cleaning import (apply_operations, convert_column, drop_duplicates,
                                    encode_categoricals, impute_missing, iqr_outliers,
                                    remove_missing)
from dataforge_sba.exploration import chart_data, summarize
from dataforge_sba.exceptions import AlgorithmError, DatasetError
from dataforge_sba.notebooks import NotebookGenerator
from dataforge_sba.playground import run_experiment, run_kmeans


def test_dataset_inspection_and_non_mutating_frame():
    source = pd.DataFrame({"value": [1.0, None, 3.0], "group": ["a", "b", "b"]})
    dataset = Dataset(source)
    assert dataset.shape == (3, 2)
    assert dataset.schema()[0]["missing"] == 1
    copy = dataset.frame; copy.loc[0, "value"] = 100
    assert dataset.frame.loc[0, "value"] == 1


def test_load_csv_and_invalid_extension(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
    assert Dataset(path).shape == (2, 2)
    bad = tmp_path / "sample.json"; bad.write_text("{}")
    with pytest.raises(DatasetError): Dataset(bad)


def test_load_xlsx_when_optional_engine_is_installed(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "sample.xlsx"
    pd.DataFrame({"value": [1, 2], "label": ["a", "b"]}).to_excel(path, index=False)
    assert Dataset(path).frame.to_dict("list") == {"value": [1, 2], "label": ["a", "b"]}


def test_cleaning_helpers():
    frame = pd.DataFrame({"x": [1.0, None, 2.0, 3.0, 4.0, 5.0, 100.0], "y": list("abcdefg")})
    assert impute_missing(frame, {"x": "median"})["x"].isna().sum() == 0
    assert int(iqr_outliers(frame, "x").sum()) == 1
    assert len(drop_duplicates(pd.concat([frame, frame.iloc[[0]]]))) == len(frame)


def test_cleaning_operations_report_effects_and_preserve_input():
    frame = pd.DataFrame({"value": [1.0, None, 3.0, 100.0], "group": ["low", "medium", "high", "low"],
                          "remove_me": [1, None, 2, 3]})
    cleaned, reports = apply_operations(frame, [
        {"type": "impute", "columns": ["value"], "strategies": {"value": "median"}},
        {"type": "convert", "column": "group", "target": "boolean"},
        {"type": "outliers", "columns": ["value"]},
    ])
    assert frame["value"].isna().sum() == 1
    assert cleaned["value"].isna().sum() == 0
    assert reports[0]["changed"] == 1
    assert reports[1]["changed"] == 4
    assert reports[2]["details"][0]["count"] == 1
    assert cleaned.shape == frame.shape


def test_cleaning_validation_missing_rows_encoding_and_type_failures():
    frame = pd.DataFrame({"number": [1.0, None, 3.0], "category": ["small", "medium", "large"]})
    assert len(remove_missing(frame, axis="rows", columns=["number"])) == 2
    with pytest.raises(ValueError, match="numeric column"):
        apply_operations(frame, [{"type": "impute", "columns": ["category"], "strategies": {"category": "mean"}}])
    with pytest.raises(ValueError, match="real order"):
        apply_operations(frame, [{"type": "encode", "columns": ["category"], "method": "ordinal"}])
    ordered = encode_categoricals(frame, ["category"], "ordinal", {"category": ["small", "medium", "large"]})
    assert ordered["category"].tolist() == [0, 1, 2]
    converted, failures = convert_column(pd.DataFrame({"value": ["1", "oops", "2"]}), "value", "numeric")
    assert failures == 1 and converted["value"].isna().sum() == 1


def test_shared_eda_charts_and_constant_summary():
    frame = pd.DataFrame({"x": [1, 1, 1, 1], "y": [2, 3, 4, 5], "group": ["a", "a", "b", None]})
    assert summarize(frame)["constant_columns"] == ["x"]
    assert chart_data(frame, "histogram", ["x"])["message"] == "All available values are the same."
    assert chart_data(frame, "boxplot", ["y"])["median"] == 3.5
    assert len(chart_data(frame, "scatter", ["x", "y"])["points"]) == 4
    assert chart_data(frame, "correlation", ["x", "y"])["matrix"][0][0] == 0
    assert chart_data(frame, "bar", ["group"])["values"] == [2, 1, 1]


def test_registry_and_models():
    frame = pd.DataFrame({"x": range(60), "z": [i % 7 for i in range(60)], "y": [i * 2 for i in range(60)]})
    result = run_algorithm("linear_regression", frame, target="y")
    assert result["metrics"]["r2"] > .99
    assert {x["id"] for x in list_algorithms()} == {"linear_regression", "logistic_regression", "kmeans", "decision_tree", "knn"}
    with pytest.raises(AlgorithmError): run_algorithm("logistic_regression", frame)


def test_all_supervised_models_with_feature_selection_and_minmax_scaling():
    frame = pd.DataFrame({"x": range(120), "unused": [i % 9 for i in range(120)],
                          "value": [i * 1.7 for i in range(120)],
                          "label": ["high" if i % 2 else "low" for i in range(120)]})
    regression = run_algorithm("linear_regression", frame, target="value", feature_columns=["x"], scaler="minmax")
    assert regression["metrics"]["rmse"] < 1e-8
    for identifier, parameters in [("logistic_regression", {}), ("decision_tree", {"max_depth": 3}),
                                   ("knn", {"n_neighbors": 3})]:
        result = run_algorithm(identifier, frame, target="label", feature_columns=["x", "unused"], scaler="minmax", **parameters)
        assert 0 <= result["metrics"]["accuracy"] <= 1
    clustered = run_algorithm("kmeans", frame, feature_columns=["x", "unused"], n_clusters=2, scaler="minmax")
    assert sum(clustered["cluster_sizes"].values()) == len(frame)
    with pytest.raises(AlgorithmError, match="at least two rows"):
        run_algorithm("logistic_regression", pd.DataFrame({"x": range(10), "label": ["single"] * 9 + ["rare"]}), target="label")


def test_supervised_imputation_is_fit_on_training_rows_only():
    from sklearn.model_selection import train_test_split
    frame = pd.DataFrame({"x": range(100), "label": ["low", "high"] * 50})
    indices = pd.Series(range(len(frame)))
    train_indices, _ = train_test_split(indices, test_size=.2, random_state=42, stratify=frame["label"])
    frame.loc[train_indices.iloc[0], "x"] = None
    expected = frame.loc[train_indices, "x"].mean()
    result = run_algorithm("logistic_regression", frame, target="label", feature_columns=["x"],
                           impute_strategies={"x": "mean"}, random_state=42)
    prep = result["model"].named_steps["preprocessing"]
    imputer = prep.named_transformers_["numeric_mean"].named_steps["imputer"]
    assert imputer.statistics_[0] == pytest.approx(expected)


def test_notebook_structure(tmp_path):
    dataset = Dataset(pd.DataFrame({"x": [1, 2, 3]}))
    path = NotebookGenerator(dataset).add_visualizations().add_algorithm("kmeans", n_clusters=2).save(tmp_path / "out.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert any(cell.cell_type == "code" for cell in notebook.cells)


def test_notebook_selection_config_and_generated_cells_execute(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    frame = pd.DataFrame({"x": range(120), "category": ["a", "b", "c"] * 40,
                          "value": [i * 2.0 for i in range(120)],
                          "label": ["yes" if i % 2 else "no" for i in range(120)],
                          "flag": ["yes" if i % 2 else "no" for i in range(120)]})
    path = tmp_path / "student data.csv"; frame.to_csv(path, index=False)
    dataset = Dataset(path)
    generator = NotebookGenerator(dataset).add_visualizations(columns=["x", "value", "category"], charts=["histogram", "boxplot", "scatter", "correlation", "bar", "frequency"])
    generator.add_cleaning(operations=[{"type": "impute", "columns": ["value"], "strategies": {"value": "median"}},
                                       {"type": "convert", "column": "flag", "target": "boolean"}])
    generator.add_algorithm("linear_regression", target="value", feature_columns=["x"], test_size=.25, random_state=7, scaler="minmax")
    generator.add_algorithm("logistic_regression", target="label", feature_columns=["x", "category"], test_size=.25, random_state=7)
    generator.add_playground({"algorithm": "linear_regression", "source": "sample", "n_samples": 60, "noise": 5, "random_state": 7})
    notebook = generator.build(); nbformat.validate(notebook)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for source in sources:
        compile(source, "<generated-notebook-cell>", "exec")
    assert any("DATA_PATH = 'student data.csv'" in source for source in sources)
    assert any("actual-vs" not in source and "MinMaxScaler" in source for source in sources)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *_args, **_kwargs: None)
    namespace = {"display": lambda *_args, **_kwargs: None}
    monkeypatch.chdir(tmp_path)
    for source in sources:
        exec(compile(source, "<generated-notebook-cell>", "exec"), namespace)
    assert len(namespace["model_comparison"]) == 2
    for algorithm in ("kmeans", "logistic_regression", "decision_tree", "knn"):
        playground_code = NotebookGenerator._playground_code({"algorithm": algorithm, "source": "sample",
            "shape": "moons", "n_samples": 60, "n_clusters": 3, "n_neighbors": 3,
            "max_depth": 3, "random_state": 13})
        exec(compile(playground_code, "<generated-playground-cell>", "exec"), namespace)
    namespace["df"] = frame
    uploaded_code = NotebookGenerator._playground_code({"algorithm": "logistic_regression", "source": "uploaded",
        "feature_x": "x", "feature_y": "value", "target": "label", "random_state": 21})
    exec(compile(uploaded_code, "<uploaded-playground-cell>", "exec"), namespace)


def test_notebook_skips_regression_for_categorical_targets():
    dataset = Dataset(pd.DataFrame({"x": [1, 2, 3]}))
    notebook = (NotebookGenerator(dataset)
                .add_algorithm("linear_regression")
                .add_algorithm("logistic_regression")
                .build())
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code" and "TARGET = None" in cell.source]
    assert len(cells) == 2
    assert "not pd.api.types.is_numeric_dtype(y)" in cells[0]
    assert "Skipped: Linear Regression needs a numeric target." in cells[0]
    assert "type_of_target(y) not in ('binary', 'multiclass')" in cells[1]
    assert "Skipped: Logistic Regression needs a categorical target," in cells[1]
    assert "classification_report(y_test, predictions" in cells[1]
    for source in cells:
        compile(source, "<generated-notebook-cell>", "exec")


def test_kmeans_playground_reproducible():
    one = run_kmeans("blobs", 80, 3, 12)
    two = run_kmeans("blobs", 80, 3, 12)
    assert one["labels"] == two["labels"]
    assert len(one["centroids"]) == 3


@pytest.mark.parametrize("algorithm", ["kmeans", "linear_regression", "logistic_regression", "decision_tree", "knn"])
def test_playground_algorithms_use_sample_data(algorithm):
    result = run_experiment(algorithm, n_samples=100, n_clusters=3)
    assert result["algorithm"] == algorithm
    assert result["points"]
    assert result["metrics"]
    if algorithm in {"logistic_regression", "decision_tree", "knn"}:
        assert len(result["boundary"]) == 35 * 35
