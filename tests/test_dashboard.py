import io

from dataforge_sba.dashboard import create_app


def test_dashboard_upload_and_routes(tmp_path):
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path), "SECRET_KEY": "test"})
    client = app.test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert b"test-workflow" in home.data
    assert b"app.js?v=0.9.1" in home.data
    assert b"app.css?v=0.9.1" in home.data
    assert client.get("/static/js/app.js?v=0.9.1").status_code == 200
    assert client.get("/static/css/app.css?v=0.9.1").status_code == 200
    assert client.get("/api/algorithms").status_code == 200
    assert client.get("/api/dataset").status_code == 404
    response = client.post("/api/upload", data={"file": (io.BytesIO(b"x,y\n1,a\n2,b\n"), "data.csv")}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.json["rows"] == 2


def test_playground_input_validation():
    app = create_app({"TESTING": True})
    response = app.test_client().post("/api/playground/kmeans", json={"n_clusters": 1})
    assert response.status_code == 400


def test_target_preflight_must_pass_before_notebook(tmp_path):
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path), "SECRET_KEY": "test"})
    client = app.test_client()
    rows = "feature,label\n" + "\n".join(f"{i}.5,{'A' if i % 2 else 'B'}" for i in range(60)) + "\n"
    upload = client.post("/api/upload", data={"file": (io.BytesIO(rows.encode()), "sample.csv")}, content_type="multipart/form-data")
    assert upload.status_code == 200
    workflow = {"visualizations": True, "visualization_types": ["histogram"], "visualization_columns": [], "drop_duplicates": False,
                "algorithms": [{"id": "logistic_regression", "target": "label", "parameters": {}}]}
    assert client.post("/api/notebook", json=workflow).status_code == 400
    result = client.post("/api/workflow/test", json=workflow)
    assert result.status_code == 200
    assert result.json["passed"] is True
    generated = client.post("/api/notebook", json=workflow)
    assert generated.status_code == 200
    assert b"TARGET = 'label'" in generated.data

    changed = {**workflow, "algorithms": [{"id": "logistic_regression", "target": "feature", "parameters": {}}]}
    assert client.post("/api/workflow/test", json=changed).json["passed"] is False
    assert client.post("/api/notebook", json=changed).status_code == 400


def test_playground_can_use_uploaded_dataset(tmp_path):
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path), "SECRET_KEY": "test"})
    client = app.test_client()
    rows = ["first,second,amount,group"]
    for i in range(80):
        group = "east" if i < 40 else "west"
        center = -3 if group == "east" else 3
        rows.append(f"{center + (i % 7) / 20},{center + (i % 5) / 20},{i / 10},{group}")
    upload = client.post("/api/upload", data={"file": (io.BytesIO(("\n".join(rows) + "\n").encode()), "play.csv")}, content_type="multipart/form-data")
    assert upload.status_code == 200

    classification = client.post("/api/playground/experiment", json={
        "algorithm": "logistic_regression", "source": "uploaded", "target": "group",
        "feature_x": "first", "feature_y": "second"})
    assert classification.status_code == 200
    assert len(classification.json["boundary"]) == 35 * 35

    regression = client.post("/api/playground/experiment", json={
        "algorithm": "linear_regression", "source": "uploaded", "target": "amount",
        "feature_x": "first"})
    assert regression.status_code == 200
    assert regression.json["metrics"]["rmse"] >= 0


def test_upload_validation_limit_sessions_and_delete(tmp_path):
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path), "SECRET_KEY": "test", "MAX_CONTENT_LENGTH": 1024})
    client_a = app.test_client(); client_b = app.test_client()
    invalid_workbook = client_a.post("/api/upload", data={"file": (io.BytesIO(b"not an xlsx"), "fake.xlsx")}, content_type="multipart/form-data")
    assert invalid_workbook.status_code == 400
    oversized = client_a.post("/api/upload", data={"file": (io.BytesIO(b"x,y\n" + b"1,2\n" * 500), "big.csv")}, content_type="multipart/form-data")
    assert oversized.status_code == 413

    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    upload = client_a.post("/api/upload", data={"file": (io.BytesIO(b"x,y\n1,2\n2,3\n"), "data.csv")}, content_type="multipart/form-data")
    assert upload.status_code == 200
    path = next(tmp_path.glob("*.csv")); original = path.read_bytes()
    assert client_b.get("/api/dataset").status_code == 404
    assert client_a.get("/api/dataset").json["rows"] == 2
    assert client_a.delete("/api/dataset").json["removed"] is True
    assert not path.exists()
    assert original == b"x,y\n1,2\n2,3\n"


def test_dashboard_cleaning_preview_apply_eda_and_playground_notebook(tmp_path):
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path), "SECRET_KEY": "test"})
    client = app.test_client()
    csv = b"x,y,label\n1,2,a\n2,,a\n3,6,b\n4,8,b\n5,10,a\n6,12,b\n7,14,a\n8,16,b\n9,18,a\n10,20,b\n11,22,a\n12,24,b\n13,26,a\n14,28,b\n15,30,a\n16,32,b\n17,34,a\n18,36,b\n19,38,a\n20,40,b\n"
    assert client.post("/api/upload", data={"file": (io.BytesIO(csv), "sample.csv")}, content_type="multipart/form-data").status_code == 200
    operations = [{"type": "impute", "columns": ["y"], "strategies": {"y": "median"}}]
    preview = client.post("/api/cleaning/preview", json=operations)
    assert preview.status_code == 200 and preview.json["after"]["missing"] == 0
    applied = client.post("/api/cleaning/apply", json=operations)
    assert applied.status_code == 200 and applied.json["reports"][0]["changed"] == 1
    assert client.post("/api/eda/chart", json={"type": "scatter", "columns": ["x", "y"]}).status_code == 200
    config = {"visualizations": False, "visualization_types": [], "visualization_columns": [],
              "drop_duplicates": False, "cleaning_operations": operations,
              "include_playground": True,
              "playground": {"algorithm": "linear_regression", "source": "uploaded", "feature_x": "x", "target": "y"},
              "algorithms": []}
    checked = client.post("/api/workflow/test", json=config)
    assert checked.status_code == 200 and checked.json["passed"] is True
    notebook = client.post("/api/notebook", json=config)
    assert notebook.status_code == 200 and b"Machine Learning Playground experiment" in notebook.data
