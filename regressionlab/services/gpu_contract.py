"""Versioned, provider-neutral contract for CUDA analysis jobs."""

from __future__ import annotations

import base64
import gzip
from io import BytesIO
import math

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data_processing import PreparedAnalysisData


SCHEMA_VERSION = "1"
MATRIX_FORMAT = "csv.gz.base64"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GPUAnalysisRequest(StrictModel):
    schema_version: str = SCHEMA_VERSION
    matrix_format: str = MATRIX_FORMAT
    matrix_payload: str
    outcome_column: str = "y"
    main_predictor_column: str = "x0"
    model_predictor_columns: list[list[str]]
    bootstrap_predictor_columns: list[str]
    bootstrap_iterations: int = Field(ge=2, le=10_000)
    random_seed: int = Field(default=42, ge=0)
    bootstrap_indices: list[list[int]] | None = None

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, value):
        if value != SCHEMA_VERSION:
            raise ValueError("Unsupported GPU request schema version.")
        return value

    @field_validator("matrix_format")
    @classmethod
    def validate_format(cls, value):
        if value != MATRIX_FORMAT:
            raise ValueError("Unsupported GPU matrix format.")
        return value


class ModelMetrics(StrictModel):
    coefficient: float | None
    standard_error: float | None
    t_value: float | None
    p_value: float | None
    ci_95: list[float | None] = Field(min_length=2, max_length=2)
    r_squared: float | None
    adjusted_r_squared: float | None
    rmse: float | None
    f_statistic: float | None
    f_p_value: float | None
    n_observations: int = Field(ge=0)
    df_residual: float | None
    condition_number: float | None

    @field_validator("*")
    @classmethod
    def reject_non_finite(cls, value):
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("GPU response metrics must be finite or null.")
        return value


class BootstrapMetrics(StrictModel):
    mean: float
    standard_error: float
    ci_95: list[float] = Field(min_length=2, max_length=2)
    samples: list[float]

    @field_validator("mean", "standard_error", "ci_95", "samples")
    @classmethod
    def reject_non_finite(cls, value):
        values = value if isinstance(value, list) else [value]
        if any(not math.isfinite(float(item)) for item in values):
            raise ValueError("GPU bootstrap metrics must be finite.")
        return value


class GPUAnalysisOutput(StrictModel):
    schema_version: str = SCHEMA_VERSION
    models: list[ModelMetrics]
    bootstrap: BootstrapMetrics
    runtime_seconds: float = Field(ge=0)
    gpu_name: str

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, value):
        if value != SCHEMA_VERSION:
            raise ValueError("Unsupported GPU response schema version.")
        return value


class RunPodResult(StrictModel):
    job_id: str | None = None
    delay_time_ms: int | None = None
    execution_time_ms: int | None = None
    end_to_end_seconds: float
    output: GPUAnalysisOutput


def build_gpu_request(
    data: PreparedAnalysisData,
    main_independent_variable: str,
    controls: list[str],
    bootstrap_iterations: int,
    random_seed: int = 42,
    bootstrap_indices: list[list[int]] | None = None,
    max_encoded_bytes: int = 15 * 1024 * 1024,
    max_decompressed_bytes: int = 256 * 1024 * 1024,
    max_bootstrap_iterations: int = 10_000,
) -> GPUAnalysisRequest:
    """Convert prepared model data to a neutral compressed GPU payload."""
    if bootstrap_iterations > min(10_000, max_bootstrap_iterations):
        raise ValueError(
            f"GPU bootstrap iterations cannot exceed {min(10_000, max_bootstrap_iterations):,}."
        )

    matrix = pd.concat([data.y.rename("y"), data.X], axis=1).reset_index(drop=True)
    if matrix.empty or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("GPU input must contain only finite numeric values.")

    predictor_mapping = {
        column: f"x{index}" for index, column in enumerate(data.X.columns)
    }
    matrix = matrix.rename(columns=predictor_mapping)

    model_predictors = []
    for index in range(len(controls) + 1):
        columns = list(data.term_map[main_independent_variable])
        for control in controls[:index]:
            columns.extend(data.term_map[control])
        model_predictors.append([predictor_mapping[column] for column in columns])

    csv_bytes = matrix.to_csv(index=False, float_format="%.17g").encode("utf-8")
    if len(csv_bytes) > max_decompressed_bytes:
        raise ValueError("Prepared GPU matrix exceeds the decompressed size limit.")
    encoded = base64.b64encode(gzip.compress(csv_bytes, compresslevel=6)).decode("ascii")
    if len(encoded.encode("ascii")) > max_encoded_bytes:
        raise ValueError("Prepared GPU payload exceeds the configured size limit.")

    return GPUAnalysisRequest(
        matrix_payload=encoded,
        main_predictor_column=predictor_mapping[main_independent_variable],
        model_predictor_columns=model_predictors,
        bootstrap_predictor_columns=model_predictors[-1],
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
        bootstrap_indices=bootstrap_indices,
    )


def decode_matrix_payload(
    request_data: GPUAnalysisRequest,
    max_encoded_bytes: int = 15 * 1024 * 1024,
    max_decompressed_bytes: int = 256 * 1024 * 1024,
) -> bytes:
    """Safely decode a worker payload while enforcing both size limits."""
    encoded = request_data.matrix_payload.encode("ascii")
    if len(encoded) > max_encoded_bytes:
        raise ValueError("Encoded GPU payload is too large.")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        with gzip.GzipFile(fileobj=BytesIO(compressed)) as archive:
            raw = archive.read(max_decompressed_bytes + 1)
    except (ValueError, OSError) as error:
        raise ValueError("GPU matrix payload is not valid gzip/Base64 data.") from error
    if len(raw) > max_decompressed_bytes:
        raise ValueError("Decompressed GPU payload is too large.")
    return raw


def merge_gpu_model_metadata(
    output: GPUAnalysisOutput,
    dependent_variable: str,
    main_independent_variable: str,
    controls: list[str],
) -> list[dict]:
    if len(output.models) != len(controls) + 1:
        raise ValueError("GPU returned an unexpected model count.")
    results = []
    for index, metrics in enumerate(output.models):
        current_controls = controls[:index]
        result = metrics.model_dump(mode="json")
        result.update({
            "model_name": f"Model {index + 1}",
            "formula": f"{dependent_variable} ~ " + " + ".join([
                main_independent_variable, *current_controls
            ]),
            "controls": current_controls,
        })
        results.append(result)
    return results
