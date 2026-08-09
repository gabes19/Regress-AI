import base64
import gzip

import pandas as pd
import pytest

from regressionlab.services.data_processing import prepare_analysis_data
from regressionlab.services.gpu_contract import (
    GPUAnalysisRequest,
    build_gpu_request,
    decode_matrix_payload,
)


def prepared_categorical_data():
    frame = pd.DataFrame({
        "outcome secret": [1.0, 2.0, 3.5, 4.0],
        "main secret": [0.5, 1.0, 1.5, 2.0],
        "group secret": ["A", "B", "A", "B"],
    })
    return prepare_analysis_data(
        frame, "outcome secret", "main secret", ["group secret"]
    )


def test_gpu_request_uses_neutral_columns_and_round_trips():
    request_data = build_gpu_request(
        prepared_categorical_data(), "main secret", ["group secret"], 25
    )
    raw = decode_matrix_payload(request_data).decode("utf-8")

    assert raw.splitlines()[0] == "y,x0,x1"
    assert "secret" not in raw
    assert request_data.model_predictor_columns == [["x0"], ["x0", "x1"]]


def test_gpu_request_rejects_non_finite_and_oversized_payloads():
    prepared = prepared_categorical_data()
    prepared.X.iloc[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        build_gpu_request(prepared, "main secret", ["group secret"], 25)

    prepared = prepared_categorical_data()
    with pytest.raises(ValueError, match="size limit"):
        build_gpu_request(
            prepared, "main secret", ["group secret"], 25, max_encoded_bytes=8
        )
    with pytest.raises(ValueError, match="decompressed size"):
        build_gpu_request(
            prepared, "main secret", ["group secret"], 25,
            max_decompressed_bytes=8,
        )


def test_gpu_payload_rejects_invalid_and_decompression_bombs():
    request_data = GPUAnalysisRequest(
        matrix_payload=base64.b64encode(gzip.compress(b"x" * 100)).decode(),
        model_predictor_columns=[["x0"]],
        bootstrap_predictor_columns=["x0"],
        bootstrap_iterations=2,
    )
    with pytest.raises(ValueError, match="Decompressed"):
        decode_matrix_payload(request_data, max_decompressed_bytes=10)

    invalid = request_data.model_copy(update={"matrix_payload": "%%%"})
    with pytest.raises(ValueError, match="not valid"):
        decode_matrix_payload(invalid)


def test_gpu_contract_rejects_unsupported_version_and_iteration_limit():
    with pytest.raises(ValueError):
        GPUAnalysisRequest(
            schema_version="2",
            matrix_payload="unused",
            model_predictor_columns=[["x0"]],
            bootstrap_predictor_columns=["x0"],
            bootstrap_iterations=2,
        )
    with pytest.raises(ValueError):
        build_gpu_request(prepared_categorical_data(), "main secret", ["group secret"], 10_001)
