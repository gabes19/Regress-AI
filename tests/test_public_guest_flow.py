from io import BytesIO
import re

import pytest

import app as app_module
from regressionlab.services.llm_summary import build_fallback_summary


def upload_csv(client, content, filename="data.csv"):
    return client.post(
        "/upload",
        data={"csv_file": (BytesIO(content), filename)},
        content_type="multipart/form-data",
    )


def assert_inline_upload_error(response, expected_message):
    assert response.status_code == 400
    assert b'<div class="error-box" role="alert"' in response.data
    assert b"CSV could not be uploaded" in response.data
    assert expected_message in response.data
    assert b"Upload CSV" in response.data


def test_upload_with_too_many_columns_renders_inline_error_and_cleans_up(
    sample_dataset_client,
    monkeypatch,
):
    client, dataset = sample_dataset_client
    upload_folder = dataset.storage_path.parent
    files_before = set(upload_folder.iterdir())
    monkeypatch.setitem(app_module.app.config, "MAX_CSV_COLUMNS", 100)
    columns = [f"column_{index}" for index in range(101)]
    content = (",".join(columns) + "\n" + ",".join("1" for _ in columns)).encode()

    response = upload_csv(client, content)

    assert_inline_upload_error(response, b"The CSV has 101 columns; the maximum is 100.")
    assert set(upload_folder.iterdir()) == files_before


@pytest.mark.parametrize(
    ("content", "filename", "message"),
    [
        (b"not csv", "notes.txt", b"Please upload a CSV file."),
        (b"", "empty.csv", b"could not be read as CSV data"),
        (b'a,b\n"1,2\n', "malformed.csv", b"could not be read as CSV data"),
        (b"\xff\xfe\xfd", "unreadable.csv", b"could not be read as CSV data"),
    ],
)
def test_invalid_uploads_render_inline_errors(
    sample_dataset_client,
    content,
    filename,
    message,
):
    client, _ = sample_dataset_client

    response = upload_csv(client, content, filename)

    assert_inline_upload_error(response, message)


def test_missing_upload_renders_inline_error(sample_dataset_client):
    client, _ = sample_dataset_client

    response = client.post("/upload", data={}, content_type="multipart/form-data")

    assert_inline_upload_error(response, b"Please choose a CSV file.")


def test_row_limit_renders_inline_error(sample_dataset_client, monkeypatch):
    client, _ = sample_dataset_client
    monkeypatch.setitem(app_module.app.config, "MAX_CSV_ROWS", 1)

    response = upload_csv(client, b"y,x\n1,2\n3,4\n")

    assert_inline_upload_error(response, b"more than 1 rows")


def test_stored_csv_read_failure_returns_to_upload_ui(sample_dataset_client):
    client, dataset = sample_dataset_client
    dataset.storage_path.write_bytes(b"\xff\xfe\xfd")

    response = client.get(f"/configure/{dataset.dataset_id}")

    assert_inline_upload_error(response, b"stored CSV could not be read")


def test_guest_header_explains_premium_features_and_keeps_upload_available(
    sample_dataset_client,
    monkeypatch,
):
    client, dataset = sample_dataset_client
    monkeypatch.setitem(app_module.app.extensions, "google_oauth", object())

    homepage = client.get("/")
    configure = client.get(f"/configure/{dataset.dataset_id}")

    assert b"Log in to access cloud GPU + LLM features" in homepage.data
    assert b"Log in to access cloud GPU + LLM features" in configure.data
    assert b'name="csv_file"' in homepage.data
    assert b'id="use_gpu"' not in configure.data
    assert b"CPU compute and a deterministic research summary" in configure.data


def test_brand_and_configure_back_link_return_to_homepage(
    sample_dataset_client,
):
    client, dataset = sample_dataset_client

    homepage = client.get("/")
    configure = client.get(f"/configure/{dataset.dataset_id}")

    brand_link = rb'<a class="brand" href="/" aria-label="RegressAI home">'
    assert re.search(brand_link, homepage.data)
    assert re.search(brand_link, configure.data)
    assert re.search(rb'<a class="button" href="/">&larr; Back</a>', configure.data)


def test_guest_analysis_never_calls_llm_or_gpu(
    sample_dataset_client,
    monkeypatch,
):
    client, dataset = sample_dataset_client

    class ForbiddenGPU:
        configured = True
        calls = 0

        def run(self, request_data):
            self.calls += 1
            raise AssertionError("Anonymous analysis called RunPod.")

    gpu = ForbiddenGPU()

    def forbidden_llm(*args, **kwargs):
        raise AssertionError("Anonymous analysis called OpenAI.")

    monkeypatch.setattr(app_module, "generate_llm_summary", forbidden_llm)
    monkeypatch.setitem(app_module.app.extensions, "google_oauth", object())
    monkeypatch.setitem(app_module.app.extensions, "runpod_client", gpu)
    monkeypatch.setitem(app_module.app.config, "RUNPOD_ENABLED", True)
    monkeypatch.setitem(app_module.app.config, "CPU_FALLBACK_MAX_WORK_UNITS", 1)

    response = client.post(
        "/analyze",
        data={
            "dataset_id": dataset.dataset_id,
            "research_question": "Does education predict wages?",
            "dependent_variable": "wage",
            "main_independent_variable": "education",
            "bootstrap_iterations": "2",
            "use_gpu": "on",
        },
    )

    assert response.status_code == 200
    assert gpu.calls == 0
    assert b"Deterministic Research Summary" in response.data
    assert b"Log in to access cloud GPU + LLM features" in response.data
    assert b"Log in to access LLM features" in response.data
    assert re.search(rb"<strong>CPU</strong>", response.data)


def test_signed_in_analysis_attempts_llm(sample_dataset_client, monkeypatch):
    client, dataset = sample_dataset_client
    calls = []

    def fake_llm(facts, **kwargs):
        calls.append(facts)
        return build_fallback_summary(facts)

    monkeypatch.setattr(app_module, "generate_llm_summary", fake_llm)
    with client.session_transaction() as session:
        session["user"] = {"id": 1, "email": "user@example.com"}

    response = client.post(
        "/analyze",
        data={
            "dataset_id": dataset.dataset_id,
            "research_question": "Does education predict wages?",
            "dependent_variable": "wage",
            "main_independent_variable": "education",
            "bootstrap_iterations": "2",
        },
    )

    assert response.status_code == 200
    assert len(calls) == 1
    assert b"LLM Research Summary" in response.data


def test_guest_datasets_and_exports_are_session_bound(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(app_module.app.config, "TESTING", True)
    monkeypatch.setitem(app_module.app.config, "WTF_CSRF_ENABLED", False)
    monkeypatch.setitem(app_module.app.config, "MIN_BOOTSTRAP_ITERATIONS", 1)
    monkeypatch.setitem(app_module.app.config, "UPLOAD_FOLDER", tmp_path / "uploads")
    monkeypatch.setitem(app_module.app.config, "REPORTS_FOLDER", tmp_path / "reports")
    monkeypatch.setitem(app_module.app.extensions, "openai_client", None)
    owner_client = app_module.app.test_client()
    other_client = app_module.app.test_client()

    upload_response = upload_csv(owner_client, b"y,x\n1,2\n2,4\n3,6\n")
    assert upload_response.status_code == 302
    dataset_url = upload_response.location
    assert owner_client.get(dataset_url).status_code == 200
    assert other_client.get(dataset_url).status_code == 404

    analysis = owner_client.post(
        "/analyze",
        data={
            "dataset_id": dataset_url.rsplit("/", 1)[-1],
            "research_question": "Does x predict y?",
            "dependent_variable": "y",
            "main_independent_variable": "x",
            "bootstrap_iterations": "2",
        },
    )
    assert analysis.status_code == 200
    token_match = re.search(rb"/export/latex/([0-9a-f]{32})", analysis.data)
    assert token_match is not None
    export_token = token_match.group(1).decode()
    assert other_client.get(f"/export/latex/{export_token}").status_code == 404


def test_guest_can_export_latex_and_pdf(
    sample_dataset_client,
    monkeypatch,
):
    client, dataset = sample_dataset_client
    analysis = client.post(
        "/analyze",
        data={
            "dataset_id": dataset.dataset_id,
            "research_question": "Does education predict wages?",
            "dependent_variable": "wage",
            "main_independent_variable": "education",
            "bootstrap_iterations": "2",
        },
    )
    token_match = re.search(rb"/export/latex/([0-9a-f]{32})", analysis.data)
    assert token_match is not None
    export_token = token_match.group(1).decode()

    latex_response = client.get(f"/export/latex/{export_token}")
    assert latex_response.status_code == 200
    assert "attachment" in latex_response.headers["Content-Disposition"]

    def fake_compile(report_dir, tex_path):
        pdf_path = report_dir / "regression_report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")
        return pdf_path

    monkeypatch.setattr(app_module, "compile_pdf_report", fake_compile)
    pdf_response = client.get(f"/export/pdf/{export_token}")
    assert pdf_response.status_code == 200
    assert "attachment" in pdf_response.headers["Content-Disposition"]
