"""HTTP boundary: validate requests and delegate work to the library."""
from flask import Blueprint, current_app, jsonify, render_template, request, send_file, session
from pathlib import Path
from werkzeug.utils import secure_filename
import hashlib
import json
import uuid
import zipfile
import io

bp = Blueprint("dashboard", __name__)
ALLOWED = {".csv", ".xlsx"}


def _raw_dataset():
    from dataforge_sba import Dataset
    path = session.get("dataset_path")
    if not path: return None
    root = Path(current_app.config["UPLOAD_FOLDER"]).resolve()
    resolved = Path(path).resolve()
    if resolved.parent != root or not resolved.is_file():
        session.pop("dataset_path", None)
        session.pop("cleaning_operations", None)
        return None
    return Dataset(path)


def _dataset():
    data = _raw_dataset()
    if data is None: return None
    operations = session.get("cleaning_operations", [])
    if operations:
        from dataforge_sba.cleaning import apply_operations
        frame, _ = apply_operations(data.frame, operations)
        from dataforge_sba import Dataset
        return Dataset(frame)
    return data


def _model_frame(data, operations, *, supervised=True):
    """Keep encoding and supervised imputation inside pipelines fitted on training rows."""
    from dataforge_sba.cleaning import apply_operations
    model_ops = [op for op in operations if op.get("type") != "encode" and not (supervised and op.get("type") == "impute")]
    return apply_operations(data.frame, model_ops)[0]


def _workflow_config(config):
    """Select the settings that must remain unchanged after a successful preflight."""
    return {"algorithms": config.get("algorithms", []),
            "drop_duplicates": bool(config.get("drop_duplicates", False)),
            "visualizations": bool(config.get("visualizations", False)),
            "visualization_types": config.get("visualization_types", []),
            "visualization_columns": config.get("visualization_columns", []),
            "include_playground": bool(config.get("include_playground", False)),
            "playground": config.get("playground") if config.get("include_playground") else None,
            "cleaning_operations": config.get("cleaning_operations", [])}


def _workflow_signature(config):
    encoded = json.dumps(_workflow_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@bp.get("/")
def home(): return render_template("index.html")


@bp.get("/api/algorithms")
def algorithms():
    from dataforge_sba.algorithms import list_algorithms
    return jsonify(list_algorithms())


@bp.post("/api/upload")
def upload():
    from dataforge_sba import Dataset
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename: return jsonify(error="Choose a CSV or XLSX file."), 400
    name = secure_filename(uploaded.filename); ext = Path(name).suffix.lower()
    if ext not in ALLOWED: return jsonify(error="Only CSV and XLSX files are supported."), 400
    uploaded.stream.seek(0, 2); file_size = uploaded.stream.tell(); uploaded.stream.seek(0)
    if file_size > current_app.config["MAX_UPLOAD_SIZE"]:
        return jsonify(error="File is larger than the 25 MB upload limit."), 413
    content = uploaded.stream.read(8192)
    uploaded.stream.seek(0)
    if ext == ".xlsx":
        if not content.startswith(b"PK"):
            return jsonify(error="The selected .xlsx file does not look like a valid Excel workbook."), 400
        # Verify the OOXML workbook signature without trusting the filename.
        full_bytes = uploaded.stream.read()
        uploaded.stream.seek(0)
        try:
            with zipfile.ZipFile(io.BytesIO(full_bytes)) as workbook:
                if "[Content_Types].xml" not in workbook.namelist() or "xl/workbook.xml" not in workbook.namelist():
                    raise ValueError("not an Excel workbook")
        except Exception:
            return jsonify(error="The selected .xlsx file is not a valid Excel workbook."), 400
    elif b"\x00" in content:
        return jsonify(error="This file does not look like a text CSV."), 400
    elif content.lstrip().lower().startswith((b"<!doctype html", b"<html", b"%pdf-", b"{", b"[")):
        return jsonify(error="This file's contents do not look like a CSV table."), 400
    target = Path(current_app.config["UPLOAD_FOLDER"]) / f"{uuid.uuid4().hex}{ext}"
    uploaded.save(target)
    try: data = Dataset(target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        return jsonify(error=str(exc)), 400
    old = session.get("dataset_path")
    if old:
        try:
            old_path = Path(old).resolve()
            if old_path.parent == Path(current_app.config["UPLOAD_FOLDER"]).resolve(): old_path.unlink(missing_ok=True)
        except OSError: pass
    session["dataset_path"] = str(target)
    session["dataset_filename"] = name
    session.pop("cleaning_operations", None)
    session.pop("validated_workflow", None)
    return jsonify({"filename": name, "rows": data.shape[0], "columns": data.shape[1], "schema": data.schema(), "model_schema": data.schema(), "preview": data.preview(8), "summary": data.explore(), "cleaning_operations": []})


@bp.get("/api/dataset")
def dataset():
    try:
        from dataforge_sba import Dataset
        data = _dataset()
        if data is None: return jsonify(error="Upload a dataset to get started."), 404
        raw = _raw_dataset()
        model_schema = Dataset(_model_frame(raw, session.get("cleaning_operations", []), supervised=True)).schema()
        return jsonify({"rows": data.shape[0], "columns": data.shape[1], "schema": data.schema(), "model_schema": model_schema, "preview": data.preview(8), "summary": data.explore(), "cleaning_operations": session.get("cleaning_operations", [])})
    except Exception as exc: return jsonify(error=str(exc)), 400


@bp.post("/api/cleaning/preview")
def cleaning_preview():
    try:
        from dataforge_sba.cleaning import apply_operations
        data = _raw_dataset()
        if data is None: return jsonify(error="Upload a dataset before previewing cleaning."), 400
        operations = request.get_json(silent=True) or []
        if not isinstance(operations, list): return jsonify(error="Cleaning selections must be a list."), 400
        before = data.frame
        after, reports = apply_operations(before, operations)
        if after.empty or not len(after.columns): return jsonify(error="These steps would leave no rows or columns. Choose fewer cleaning operations."), 400
        from dataforge_sba import Dataset
        summary = Dataset(after).explore()
        return jsonify({"before": {"rows": len(before), "columns": len(before.columns),
                                    "missing": int(before.isna().sum().sum()), "duplicates": int(before.duplicated().sum())},
                        "after": {"rows": len(after), "columns": len(after.columns),
                                   "missing": int(after.isna().sum().sum()), "duplicates": int(after.duplicated().sum())},
                        "reports": reports, "schema": Dataset(after).schema(), "summary": summary,
                        "preview": Dataset(after).preview(8)})
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@bp.post("/api/cleaning/apply")
def cleaning_apply():
    try:
        from dataforge_sba.cleaning import apply_operations
        data = _raw_dataset()
        if data is None: return jsonify(error="Upload a dataset before applying cleaning."), 400
        operations = request.get_json(silent=True) or []
        if not isinstance(operations, list): return jsonify(error="Cleaning selections must be a list."), 400
        frame, reports = apply_operations(data.frame, operations)
        if frame.empty or not len(frame.columns): return jsonify(error="These steps would leave no rows or columns. Choose fewer cleaning operations."), 400
        from dataforge_sba import Dataset
        cleaned = Dataset(frame)
        model_frame = _model_frame(data, operations, supervised=True)
        session["cleaning_operations"] = operations
        session.pop("validated_workflow", None)
        return jsonify({"rows": cleaned.shape[0], "columns": cleaned.shape[1], "schema": cleaned.schema(), "model_schema": Dataset(model_frame).schema(),
                        "summary": cleaned.explore(), "preview": cleaned.preview(8), "reports": reports})
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@bp.delete("/api/dataset")
def delete_dataset():
    """Remove this browser session's upload and reset its dataset selections."""
    path = session.get("dataset_path")
    if path:
        root = Path(current_app.config["UPLOAD_FOLDER"]).resolve()
        resolved = Path(path).resolve()
        if resolved.parent == root:
            resolved.unlink(missing_ok=True)
    for key in ("dataset_path", "dataset_filename", "cleaning_operations", "validated_workflow"):
        session.pop(key, None)
    return jsonify({"removed": True})


@bp.post("/api/eda/chart")
def eda_chart():
    try:
        from dataforge_sba.exploration import chart_data
        data = _dataset()
        if data is None: return jsonify(error="Upload a dataset before creating a chart."), 400
        args = request.get_json(silent=True) or {}
        return jsonify(chart_data(data.frame, str(args.get("type", "")), args.get("columns") or []))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@bp.post("/api/notebook")
def notebook():
    try:
        from dataforge_sba.notebooks import NotebookGenerator
        data = _raw_dataset()
        if data is None: return jsonify(error="Upload a dataset first."), 400
        config = request.get_json(silent=True) or {}; generator = NotebookGenerator(data, dataset_name=session.get("dataset_filename"))
        if session.get("validated_workflow") != _workflow_signature(config):
            return jsonify(error="Test this exact algorithm and target selection successfully before generating the notebook."), 400
        if config.get("visualizations"):
            generator.add_visualizations(config.get("visualization_columns"), config.get("visualization_types"))
        if config.get("drop_duplicates") or config.get("cleaning_operations"):
            generator.add_cleaning(drop_duplicates=bool(config.get("drop_duplicates")),
                                   operations=config.get("cleaning_operations", []))
        from dataforge_sba.cleaning import model_imputation_strategies
        impute_strategies = model_imputation_strategies(config.get("cleaning_operations", []))
        for item in config.get("algorithms", []):
            generator.add_algorithm(item["id"], target=item.get("target"),
                                    impute_strategies=impute_strategies, **item.get("parameters", {}))
        if config.get("include_playground"):
            settings = dict(config.get("playground") or {})
            settings["impute_strategies"] = impute_strategies
            generator.add_playground(settings)
        out = Path(current_app.config["UPLOAD_FOLDER"]) / f"analysis-{uuid.uuid4().hex}.ipynb"
        generator.save(out)
        response = send_file(out, as_attachment=True, download_name="dataforge_analysis.ipynb", mimetype="application/x-ipynb+json")
        response.call_on_close(lambda: out.unlink(missing_ok=True))
        return response
    except Exception as exc: return jsonify(error=str(exc)), 400


@bp.post("/api/workflow/test")
def test_workflow():
    """Run the selected library algorithms on the uploaded data before notebook creation."""
    try:
        from dataforge_sba.algorithms import list_algorithms, run_algorithm
        from dataforge_sba.cleaning import apply_operations, drop_duplicates
        data = _raw_dataset()
        if data is None: return jsonify(error="Upload a dataset before testing algorithms."), 400
        config = request.get_json(silent=True) or {}
        selected = config.get("algorithms", [])
        include_playground = bool(config.get("include_playground"))
        has_other_sections = bool(config.get("visualizations") or config.get("drop_duplicates")
                                  or config.get("cleaning_operations"))
        if not isinstance(selected, list) or (not selected and not include_playground and not has_other_sections):
            return jsonify(error="Select an algorithm, an EDA chart, a cleaning step, or the Playground experiment."), 400
        specs = {item["id"]: item for item in list_algorithms()}
        cleaning_ops = config.get("cleaning_operations", [])
        has_supervised = any(isinstance(item, dict) and item.get("id") != "kmeans" for item in selected)
        playground_settings = config.get("playground") or {}
        has_supervised = has_supervised or (include_playground and playground_settings.get("algorithm") != "kmeans")
        # Category encoders in the model pipelines are fitted after the split.
        from dataforge_sba.cleaning import model_imputation_strategies
        impute_strategies = model_imputation_strategies(cleaning_ops)
        model_ops = [op for op in cleaning_ops if op.get("type") != "encode" and not (has_supervised and op.get("type") == "impute")]
        frame, _ = apply_operations(data.frame, model_ops)
        if config.get("drop_duplicates"):
            frame = drop_duplicates(frame)
        results = []
        if config.get("cleaning_operations") or config.get("drop_duplicates"):
            results.append({"id": "cleaning", "ok": True, "message": "Selected cleaning operations passed validation."})
        if config.get("visualizations"):
            try:
                from dataforge_sba.exploration import validate_visualization_selection
                chart_frame, _ = apply_operations(data.frame, cleaning_ops)
                if config.get("drop_duplicates"): chart_frame = drop_duplicates(chart_frame)
                validate_visualization_selection(chart_frame, config.get("visualization_types", []),
                                                 config.get("visualization_columns", []))
                results.append({"id": "visualizations", "ok": True, "message": "Selected EDA charts have compatible columns."})
            except Exception as exc:
                results.append({"id": "visualizations", "ok": False, "message": str(exc)})
        for item in selected:
            if not isinstance(item, dict) or item.get("id") not in specs:
                results.append({"id": "unknown", "ok": False, "message": "Choose a valid algorithm."})
                continue
            identifier = item["id"]; spec = specs[identifier]; target = item.get("target")
            if spec["supervised"] and not target:
                results.append({"id": identifier, "ok": False, "message": "Choose a target column before testing."})
                continue
            if target and target not in frame.columns:
                results.append({"id": identifier, "ok": False, "message": "The target column is not in the uploaded dataset."})
                continue
            try:
                result = run_algorithm(identifier, frame, target=target, impute_strategies=impute_strategies,
                                       **(item.get("parameters") or {}))
                details = result.get("metrics", {})
                if identifier == "kmeans":
                    details = {"silhouette": result.get("silhouette_score"),
                               "clusters": len(result.get("cluster_sizes", {}))}
                results.append({"id": identifier, "ok": True, "message": "Test passed on your uploaded dataset.", "metrics": details})
            except Exception as exc:
                results.append({"id": identifier, "ok": False, "message": str(exc)})
        if include_playground:
            try:
                from dataforge_sba.playground import run_experiment
                settings = config.get("playground") or {}
                source = settings.get("source", "sample")
                run_experiment(settings.get("algorithm", "kmeans"), source=source,
                    frame=frame if source == "uploaded" else None,
                    shape=settings.get("shape", "blobs"), n_samples=int(settings.get("n_samples", 300)),
                    n_clusters=int(settings.get("n_clusters", 3)), scale=bool(settings.get("scale", True)),
                    noise=float(settings.get("noise", 15)), max_depth=int(settings.get("max_depth", 4)),
                    n_neighbors=int(settings.get("n_neighbors", 5)), target=settings.get("target"),
                    feature_x=settings.get("feature_x"), feature_y=settings.get("feature_y"),
                    impute_strategies=impute_strategies)
                results.append({"id": "playground", "ok": True, "message": "Playground experiment passed."})
            except Exception as exc:
                results.append({"id": "playground", "ok": False, "message": str(exc)})
        passed = bool(results) and all(result["ok"] for result in results)
        if passed:
            session["validated_workflow"] = _workflow_signature(config)
        else:
            session.pop("validated_workflow", None)
        return jsonify({"passed": passed, "results": results})
    except Exception as exc:
        session.pop("validated_workflow", None)
        return jsonify(error=str(exc)), 400


@bp.post("/api/playground/kmeans")
def playground():
    try:
        from dataforge_sba.playground import run_kmeans
        args = request.get_json(silent=True) or {}
        n = int(args.get("n_samples", 300)); k = int(args.get("n_clusters", 3)); seed = int(args.get("random_state", 42))
        if not 50 <= n <= 1000: return jsonify(error="Sample size must be between 50 and 1000."), 400
        if not 2 <= k <= min(12, n): return jsonify(error="Choose between 2 and 12 clusters."), 400
        return jsonify(run_kmeans(str(args.get("shape", "blobs")), n, k, seed, bool(args.get("scale", True))))
    except Exception as exc: return jsonify(error=str(exc)), 400


@bp.post("/api/playground/experiment")
def playground_experiment():
    """Run an interactive model experiment on sample data or this session's upload."""
    try:
        from dataforge_sba.playground import run_experiment
        args = request.get_json(silent=True) or {}
        source = str(args.get("source", "sample"))
        frame = None
        if source == "uploaded":
            data = _raw_dataset()
            if data is None: return jsonify(error="Upload a dataset before choosing uploaded CSV data."), 400
            frame = _model_frame(data, session.get("cleaning_operations", []),
                                 supervised=str(args.get("algorithm", "kmeans")) != "kmeans")
        from dataforge_sba.cleaning import model_imputation_strategies
        impute_strategies = model_imputation_strategies(session.get("cleaning_operations", []))
        n_samples = int(args.get("n_samples", 300))
        n_clusters = int(args.get("n_clusters", 3))
        neighbors = int(args.get("n_neighbors", 5))
        depth = int(args.get("max_depth", 4))
        if source == "sample" and not 50 <= n_samples <= 1000:
            return jsonify(error="Sample size must be between 50 and 1000."), 400
        if not 2 <= n_clusters <= 12: return jsonify(error="Choose between 2 and 12 clusters."), 400
        if not 1 <= neighbors <= 50: return jsonify(error="Choose between 1 and 50 neighbors."), 400
        if not 1 <= depth <= 20: return jsonify(error="Tree depth must be between 1 and 20."), 400
        result = run_experiment(
            algorithm=str(args.get("algorithm", "kmeans")), source=source, frame=frame,
            shape=str(args.get("shape", "blobs")), n_samples=n_samples,
            random_state=int(args.get("random_state", 42)),
            noise=float(args.get("noise", 15)), n_clusters=n_clusters,
            scale=bool(args.get("scale", True)), max_depth=depth,
            n_neighbors=neighbors, target=args.get("target"),
            feature_x=args.get("feature_x"), feature_y=args.get("feature_y"),
            impute_strategies=impute_strategies,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify(error=str(exc)), 400
