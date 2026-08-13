from io import BytesIO
import re

import app as app_module
import pytest

from regressionlab.services.dataset_service import (
    DatasetNotFoundError,
    load_dataset,
)


DATASET_LOCATION_PATTERN = re.compile(r"/configure/([0-9a-f]{32})$")


def test_upload_redirects_to_id_based_configuration(
    sample_dataset_client,
):
    client, _ = sample_dataset_client

    response = client.post(
        "/upload",
        data={
            "csv_file": (
                BytesIO(b"outcome,predictor\n1,2\n3,4\n"),
                "../../unsafe name.csv",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 302
    location_match = DATASET_LOCATION_PATTERN.search(response.location)
    assert location_match is not None

    dataset_id = location_match.group(1)
    dataset = load_dataset(
        dataset_id,
        upload_folder=app_module.app.config["UPLOAD_FOLDER"],
    )
    assert dataset.original_filename == "unsafe_name.csv"
    assert dataset.storage_path.name == f"{dataset_id}.csv"
    assert dataset.storage_path.parent == app_module.app.config[
        "UPLOAD_FOLDER"
    ].resolve()


def test_configuration_uses_dataset_id_not_storage_path(
    sample_dataset_client,
):
    client, dataset = sample_dataset_client

    response = client.get(f"/configure/{dataset.dataset_id}")

    assert response.status_code == 200
    assert dataset.original_filename.encode() in response.data
    assert (
        f'name="dataset_id" value="{dataset.dataset_id}"'.encode()
        in response.data
    )
    assert str(dataset.storage_path).encode() not in response.data


def test_upload_rejects_non_csv_without_creating_dataset(
    sample_dataset_client,
):
    client, dataset = sample_dataset_client
    upload_folder = dataset.storage_path.parent
    files_before = set(upload_folder.iterdir())

    response = client.post(
        "/upload",
        data={"csv_file": (BytesIO(b"not csv"), "notes.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"Please upload a CSV file" in response.data
    assert set(upload_folder.iterdir()) == files_before


def test_analyze_rejects_path_traversal_before_reading_csv(
    sample_dataset_client,
    monkeypatch,
):
    client, _ = sample_dataset_client

    def unexpected_csv_read(*args, **kwargs):
        raise AssertionError("Untrusted dataset ID reached pandas.")

    monkeypatch.setattr(app_module.pd, "read_csv", unexpected_csv_read)

    response = client.post(
        "/analyze",
        data={
            "dataset_id": "../../wage_education_sample",
            "research_question": "Question",
            "dependent_variable": "wage",
            "main_independent_variable": "education",
            "bootstrap_iterations": "100",
        },
    )

    assert response.status_code == 404
    assert b"Dataset not found" in response.data


def test_analyze_rejects_unknown_dataset_id(sample_dataset_client):
    client, _ = sample_dataset_client

    response = client.post(
        "/analyze",
        data={
            "dataset_id": "0" * 32,
            "research_question": "Question",
            "dependent_variable": "wage",
            "main_independent_variable": "education",
            "bootstrap_iterations": "100",
        },
    )

    assert response.status_code == 404
    assert b"Dataset not found" in response.data


def test_successful_analysis_deletes_csv_and_metadata(
    sample_dataset_client,
):
    client, dataset = sample_dataset_client
    metadata_path = dataset.storage_path.with_suffix(".json")

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
    assert not dataset.storage_path.exists()
    assert not metadata_path.exists()
    assert b"Uploaded CSV deleted from RegressAI storage" in response.data
    with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
        load_dataset(
            dataset.dataset_id,
            upload_folder=app_module.app.config["UPLOAD_FOLDER"],
        )


def test_failed_analysis_keeps_csv_for_correcting_configuration(
    sample_dataset_client,
):
    client, dataset = sample_dataset_client
    metadata_path = dataset.storage_path.with_suffix(".json")

    response = client.post(
        "/analyze",
        data={
            "dataset_id": dataset.dataset_id,
            "research_question": "Does wage predict itself?",
            "dependent_variable": "wage",
            "main_independent_variable": "wage",
            "bootstrap_iterations": "2",
        },
    )

    assert response.status_code == 400
    assert dataset.storage_path.exists()
    assert metadata_path.exists()
    assert client.get(f"/configure/{dataset.dataset_id}").status_code == 200


def test_analysis_fails_closed_when_csv_deletion_cannot_be_confirmed(
    sample_dataset_client,
    monkeypatch,
):
    client, dataset = sample_dataset_client

    def fail_deletion(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(app_module, "delete_dataset", fail_deletion)
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

    assert response.status_code == 500
    assert b"Uploaded CSV deleted from RegressAI storage" not in response.data
    assert dataset.storage_path.exists()


def test_sample_dataset_enters_same_id_lifecycle(
    sample_dataset_client,
):
    client, _ = sample_dataset_client

    response = client.post("/sample/wage-education")

    assert response.status_code == 302
    location_match = DATASET_LOCATION_PATTERN.search(response.location)
    assert location_match is not None
    assert client.get(response.location).status_code == 200
