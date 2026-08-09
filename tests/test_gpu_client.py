import json
import logging

import pytest

from regressionlab.services.gpu_client import GPUServiceError, RunPodClient
from regressionlab.services.gpu_contract import GPUAnalysisRequest


def request_data(secret="SECRET_MATRIX_PAYLOAD"):
    return GPUAnalysisRequest(
        matrix_payload=secret,
        model_predictor_columns=[["x0"]],
        bootstrap_predictor_columns=["x0"],
        bootstrap_iterations=2,
    )


def output_envelope():
    return {
        "id": "job-123", "status": "COMPLETED", "delayTime": 20,
        "executionTime": 1001,
        "output": {
            "schema_version": "1",
            "models": [{
                "coefficient": 1.0, "standard_error": .1, "t_value": 10,
                "p_value": .001, "ci_95": [.8, 1.2], "r_squared": .8,
                "adjusted_r_squared": .75, "rmse": .3, "f_statistic": 100,
                "f_p_value": .001, "n_observations": 10, "df_residual": 8,
                "condition_number": 4,
            }],
            "bootstrap": {
                "mean": 1, "standard_error": .1, "ci_95": [.8, 1.2],
                "samples": [.9, 1.1],
            },
            "runtime_seconds": .2, "gpu_name": "A4000",
        },
    }


class FakeResponse:
    def __init__(self, document):
        self.document = document

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.document).encode()


def test_runpod_client_makes_one_bounded_request_and_logs_no_payload(monkeypatch, caplog):
    calls = []

    def fake_urlopen(http_request, timeout):
        calls.append((http_request, timeout))
        return FakeResponse(output_envelope())

    monkeypatch.setattr("regressionlab.services.gpu_client.urlopen", fake_urlopen)
    client = RunPodClient("endpoint", "API_SECRET", wait_milliseconds=90_000, timeout_seconds=100)
    with caplog.at_level(logging.INFO):
        result = client.run(request_data())

    assert len(calls) == 1
    http_request, timeout = calls[0]
    body = json.loads(http_request.data)
    assert http_request.full_url.endswith("/runsync?wait=90000")
    assert timeout == 100
    assert body["policy"]["executionTimeout"] == 60_000
    assert result.execution_time_ms == 1001
    assert "estimated_cost_usd=0.0004" in caplog.text
    assert "SECRET_MATRIX_PAYLOAD" not in caplog.text
    assert "API_SECRET" not in caplog.text


def test_runpod_client_wraps_malformed_output(monkeypatch):
    monkeypatch.setattr(
        "regressionlab.services.gpu_client.urlopen",
        lambda *args, **kwargs: FakeResponse({"id": "job", "status": "COMPLETED", "output": {}}),
    )
    with pytest.raises(GPUServiceError, match="invalid response"):
        RunPodClient("endpoint", "key").run(request_data("not-logged"))


def test_runpod_client_missing_credentials_fails_without_network(monkeypatch):
    monkeypatch.setattr(
        "regressionlab.services.gpu_client.urlopen",
        lambda *args, **kwargs: pytest.fail("network should not be called"),
    )
    with pytest.raises(GPUServiceError, match="not configured"):
        RunPodClient(None, None).run(request_data())
