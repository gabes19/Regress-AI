from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path

import pytest
from flask import Flask
from werkzeug.datastructures import FileStorage

import app as app_module
from regressionlab.services.data_processing import validate_csv_shape
from regressionlab.services.dataset_service import (
    DatasetNotFoundError,
    cleanup_expired_datasets,
    load_dataset,
    store_uploaded_dataset,
)
from regressionlab.services.export_report import (
    ExportNotFoundError,
    cleanup_expired_exports,
    compile_pdf_report,
    load_export_payload,
    store_export_payload,
)


def uploaded_csv(content=b"y,x\n1,2\n3,4\n", filename="data.csv"):
    return FileStorage(stream=BytesIO(content), filename=filename)


def test_health_check_is_shallow_and_has_security_headers(monkeypatch):
    monkeypatch.setitem(app_module.app.config, "TESTING", True)
    response = app_module.app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_upload_requires_csrf_token(tmp_path, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "TESTING", True)
    monkeypatch.setitem(app_module.app.config, "WTF_CSRF_ENABLED", True)
    monkeypatch.setitem(app_module.app.config, "AUTH_REQUIRED", False)
    monkeypatch.setitem(app_module.app.config, "UPLOAD_FOLDER", tmp_path)

    response = app_module.app.test_client().post(
        "/upload",
        data={"csv_file": (BytesIO(b"y,x\n1,2\n"), "data.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"Refresh the page and try again" in response.data


def test_upload_size_limit_returns_recoverable_error(tmp_path, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "TESTING", True)
    monkeypatch.setitem(app_module.app.config, "WTF_CSRF_ENABLED", False)
    monkeypatch.setitem(app_module.app.config, "AUTH_REQUIRED", False)
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 128)
    monkeypatch.setitem(app_module.app.config, "UPLOAD_FOLDER", tmp_path)

    response = app_module.app.test_client().post(
        "/upload",
        data={"csv_file": (BytesIO(b"x" * 1024), "data.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    assert b"Upload too large" in response.data


def test_upload_rate_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "TESTING", False)
    monkeypatch.setitem(app_module.app.config, "WTF_CSRF_ENABLED", False)
    monkeypatch.setitem(app_module.app.config, "AUTH_REQUIRED", False)
    monkeypatch.setitem(app_module.app.config, "UPLOAD_RATE_LIMIT", "1 per hour")
    monkeypatch.setitem(app_module.app.config, "UPLOAD_FOLDER", tmp_path)
    app_module.limiter.reset()
    client = app_module.app.test_client()

    try:
        first = client.post(
            "/upload",
            data={"csv_file": (BytesIO(b"y,x\n1,2\n"), "first.csv")},
            content_type="multipart/form-data",
        )
        second = client.post(
            "/upload",
            data={"csv_file": (BytesIO(b"y,x\n1,2\n"), "second.csv")},
            content_type="multipart/form-data",
        )
    finally:
        app_module.limiter.reset()

    assert first.status_code == 302
    assert second.status_code == 429
    assert b"Too many requests" in second.data


def test_protected_routes_require_sign_in(monkeypatch):
    monkeypatch.setitem(app_module.app.config, "TESTING", False)
    monkeypatch.setitem(app_module.app.config, "AUTH_REQUIRED", True)
    monkeypatch.setitem(app_module.app.extensions, "google_oauth", object())

    response = app_module.app.test_client().get("/configure/" + "0" * 32)

    assert response.status_code == 302
    assert response.location.endswith("/login")


def test_dataset_owner_is_enforced(tmp_path):
    dataset = store_uploaded_dataset(uploaded_csv(), tmp_path, owner_id=7)

    assert load_dataset(
        dataset.dataset_id,
        tmp_path,
        owner_id=7,
        enforce_owner=True,
    ).owner_id == 7
    with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
        load_dataset(
            dataset.dataset_id,
            tmp_path,
            owner_id=8,
            enforce_owner=True,
        )


def test_export_owner_is_enforced(tmp_path):
    token = store_export_payload(
        {"created_at": datetime.now(timezone.utc).isoformat()},
        tmp_path,
        owner_id=7,
    )

    assert load_export_payload(
        token, tmp_path, owner_id=7, enforce_owner=True
    )["owner_id"] == 7
    with pytest.raises(ExportNotFoundError, match="not found"):
        load_export_payload(
            token, tmp_path, owner_id=8, enforce_owner=True
        )


def test_csv_shape_limits_rows_and_columns(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    with pytest.raises(ValueError, match="maximum is 2"):
        validate_csv_shape(csv_path, max_rows=10, max_columns=2)
    with pytest.raises(ValueError, match="more than 1 rows"):
        validate_csv_shape(csv_path, max_rows=1, max_columns=3)


def test_retention_cleanup_removes_only_expired_artifacts(tmp_path):
    uploads = tmp_path / "uploads"
    reports = tmp_path / "reports"
    now = datetime.now(timezone.utc)
    old = now - timedelta(hours=25)

    old_dataset = store_uploaded_dataset(uploaded_csv(), uploads)
    new_dataset = store_uploaded_dataset(uploaded_csv(), uploads)
    metadata_path = uploads / f"{old_dataset.dataset_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["created_at"] = old.isoformat()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    old_token = store_export_payload(
        {"created_at": old.isoformat()}, reports
    )
    new_token = store_export_payload(
        {"created_at": now.isoformat()}, reports
    )
    old_report_dir = reports / old_token
    old_report_dir.mkdir()
    (old_report_dir / "artifact.txt").write_text("old", encoding="utf-8")

    assert cleanup_expired_datasets(uploads, 24, now=now) == 1
    assert cleanup_expired_exports(reports, 24, now=now) == 1
    assert not old_dataset.storage_path.exists()
    assert new_dataset.storage_path.exists()
    assert not (reports / "export_payloads" / f"{old_token}.json").exists()
    assert not old_report_dir.exists()
    assert (reports / "export_payloads" / f"{new_token}.json").exists()


def test_pdf_command_is_compatible_with_tex_live(tmp_path, monkeypatch):
    tex_path = tmp_path / "report.tex"
    tex_path.write_text("test", encoding="utf-8")
    captured = {}

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        (tmp_path / "regression_report.pdf").write_bytes(b"pdf")
        return Result()

    monkeypatch.setattr(
        "regressionlab.services.export_report.subprocess.run", fake_run
    )

    assert compile_pdf_report(tmp_path, tex_path).exists()
    assert "-enable-installer" not in captured["command"]


def test_production_configuration_fails_closed(monkeypatch):
    flask_app = Flask("production-check")
    flask_app.config.update(
        IS_PRODUCTION=True,
        SESSION_COOKIE_SECURE=False,
        PROXY_FIX_ENABLED=False,
        TRUSTED_HOSTS=None,
        AUTH_REQUIRED=True,
        GOOGLE_CLIENT_ID=None,
        GOOGLE_CLIENT_SECRET=None,
    )
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.delenv("DATA_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="Unsafe production configuration"):
        app_module.validate_production_configuration(flask_app)


def test_production_image_includes_static_assets():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY static ./static" in dockerfile


def test_logo_static_asset_is_served(monkeypatch):
    monkeypatch.setitem(app_module.app.config, "TESTING", True)

    response = app_module.app.test_client().get("/static/logo.png")

    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.data.startswith(b"\x89PNG\r\n\x1a\n")
