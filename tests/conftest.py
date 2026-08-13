from pathlib import Path

import pytest

from app import GUEST_OWNER_SESSION_KEY, app
from regressionlab.services.dataset_service import store_existing_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "wage_education_sample.csv"


@pytest.fixture
def sample_dataset_client(tmp_path, monkeypatch):
    upload_folder = tmp_path / "uploads"
    reports_folder = tmp_path / "reports"
    monkeypatch.setitem(app.config, "TESTING", True)
    monkeypatch.setitem(app.config, "WTF_CSRF_ENABLED", False)
    monkeypatch.setitem(app.config, "MIN_BOOTSTRAP_ITERATIONS", 1)
    monkeypatch.setitem(app.config, "UPLOAD_FOLDER", upload_folder)
    monkeypatch.setitem(app.config, "REPORTS_FOLDER", reports_folder)
    monkeypatch.setitem(app.extensions, "openai_client", None)

    client = app.test_client()
    guest_id = "a" * 32
    with client.session_transaction() as session:
        session[GUEST_OWNER_SESSION_KEY] = guest_id

    dataset = store_existing_dataset(
        SAMPLE_CSV,
        upload_folder=upload_folder,
        owner_id=f"guest:{guest_id}",
    )

    return client, dataset
