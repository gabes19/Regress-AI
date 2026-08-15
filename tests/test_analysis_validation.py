import re


def test_analysis_value_error_is_shown_on_configuration_page(
    sample_dataset_client,
):
    client, dataset = sample_dataset_client

    response = client.post(
        "/analyze",
        data={
            "dataset_id": dataset.dataset_id,
            "research_question": "Does wage predict itself?",
            "dependent_variable": "wage",
            "main_independent_variable": "wage",
            "controls": ["experience", "gender"],
            "bootstrap_iterations": "100",
        },
    )

    assert response.status_code == 400
    assert b"Analysis could not run" in response.data
    assert (
        b"The dependent and main independent variable must be different."
        in response.data
    )
    assert b"Does wage predict itself?" in response.data
    assert re.search(rb'value="experience"\s+checked', response.data)
    assert re.search(rb'value="gender"\s+checked', response.data)


def test_invalid_bootstrap_iterations_are_shown_on_configuration_page(
    sample_dataset_client,
):
    client, dataset = sample_dataset_client

    response = client.post(
        "/analyze",
        data={
            "dataset_id": dataset.dataset_id,
            "research_question": "Does education predict wages?",
            "dependent_variable": "wage",
            "main_independent_variable": "education",
            "bootstrap_iterations": "not-a-number",
        },
    )

    assert response.status_code == 400
    assert b"Bootstrap iterations must be a whole number." in response.data


def test_gpu_checkbox_is_always_available_to_signed_in_users(
    sample_dataset_client,
):
    client, dataset = sample_dataset_client
    with client.session_transaction() as session:
        session["user"] = {"id": 1, "email": "user@example.com"}

    response = client.get(f"/configure/{dataset.dataset_id}")

    assert response.status_code == 200
    panel = re.search(
        rb'<fieldset\s+id="gpu-opt-in-panel"(?P<attributes>[^>]*)>',
        response.data,
    )
    assert panel and b"hidden" not in panel.group("attributes")
    checkbox = re.search(rb'<input[^>]+id="use_gpu"[^>]*>', response.data)
    assert checkbox and b"disabled" not in checkbox.group(0)
    assert b"Use cloud GPU for this analysis" in response.data


def test_gpu_checkbox_selection_is_preserved_after_validation_error(
    sample_dataset_client,
):
    client, dataset = sample_dataset_client
    with client.session_transaction() as session:
        session["user"] = {"id": 1, "email": "user@example.com"}

    response = client.post(
        "/analyze",
        data={
            "dataset_id": dataset.dataset_id,
            "research_question": "Does wage predict itself?",
            "dependent_variable": "wage",
            "main_independent_variable": "wage",
            "bootstrap_iterations": "2500",
            "use_gpu": "on",
        },
    )

    assert response.status_code == 400
    panel = re.search(
        rb'<fieldset\s+id="gpu-opt-in-panel"(?P<attributes>[^>]*)>',
        response.data,
    )
    assert panel and b"hidden" not in panel.group("attributes")
    assert re.search(rb'<input[^>]+id="use_gpu"[^>]+checked[^>]*>', response.data)
