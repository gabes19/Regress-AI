"""CuPy implementation used only inside the RunPod CUDA worker."""

from __future__ import annotations

from io import BytesIO
import math
from time import perf_counter

import cupy as cp
import numpy as np
import pandas as pd
from scipy import stats

from regressionlab.services.gpu_contract import (
    BootstrapMetrics,
    GPUAnalysisOutput,
    GPUAnalysisRequest,
    ModelMetrics,
    decode_matrix_payload,
)


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _fit_ols_gpu(X, y, main_index):
    n_observations, n_parameters = X.shape
    x_pinv = cp.linalg.pinv(X)
    beta = x_pinv @ y
    residuals = y - X @ beta
    df_residual = n_observations - n_parameters
    sse = cp.sum(residuals ** 2)
    centered = y - cp.mean(y)
    tss = cp.sum(centered ** 2)

    if df_residual > 0:
        mse_resid = sse / df_residual
        covariance = mse_resid * (x_pinv @ x_pinv.T)
        standard_errors = cp.sqrt(cp.maximum(cp.diag(covariance), 0))
        t_values = beta / standard_errors
        beta_cpu = cp.asnumpy(beta)
        se_cpu = cp.asnumpy(standard_errors)
        t_cpu = cp.asnumpy(t_values)
        p_values = 2 * stats.t.sf(np.abs(t_cpu), df_residual)
        critical = stats.t.ppf(0.975, df_residual)
        ci_lower = beta_cpu - critical * se_cpu
        ci_upper = beta_cpu + critical * se_cpu
        rmse = cp.sqrt(mse_resid)
    else:
        beta_cpu = cp.asnumpy(beta)
        se_cpu = np.full(n_parameters, np.nan)
        t_cpu = np.full(n_parameters, np.nan)
        p_values = np.full(n_parameters, np.nan)
        ci_lower = np.full(n_parameters, np.nan)
        ci_upper = np.full(n_parameters, np.nan)
        rmse = cp.asarray(cp.nan)

    r_squared = 1 - sse / tss if float(tss.get()) > 0 else cp.asarray(cp.nan)
    if df_residual > 0 and n_observations > 1:
        adjusted = 1 - (1 - r_squared) * (n_observations - 1) / df_residual
    else:
        adjusted = cp.asarray(cp.nan)

    predictor_count = n_parameters - 1
    if predictor_count > 0 and df_residual > 0 and float(sse.get()) > 0:
        explained = tss - sse
        f_statistic = (explained / predictor_count) / (sse / df_residual)
        f_value = _finite(f_statistic.get())
        f_p_value = _finite(stats.f.sf(f_value, predictor_count, df_residual)) if f_value is not None else None
    else:
        f_value = None
        f_p_value = None

    return ModelMetrics(
        coefficient=_finite(beta_cpu[main_index]),
        standard_error=_finite(se_cpu[main_index]),
        t_value=_finite(t_cpu[main_index]),
        p_value=_finite(p_values[main_index]),
        ci_95=[_finite(ci_lower[main_index]), _finite(ci_upper[main_index])],
        r_squared=_finite(r_squared.get()),
        adjusted_r_squared=_finite(adjusted.get()),
        rmse=_finite(rmse.get()),
        f_statistic=f_value,
        f_p_value=f_p_value,
        n_observations=n_observations,
        df_residual=_finite(df_residual),
        condition_number=_finite(cp.linalg.cond(X).get()),
    )


def _bootstrap_gpu(X, y, main_index, iterations, random_seed, explicit_indices, batch_size=128):
    n_observations = X.shape[0]
    if explicit_indices is not None:
        indices = np.asarray(explicit_indices, dtype=np.int64)
        if indices.shape != (iterations, n_observations):
            raise ValueError("Explicit bootstrap indices have the wrong shape.")
        if indices.min(initial=0) < 0 or indices.max(initial=0) >= n_observations:
            raise ValueError("Explicit bootstrap indices are out of range.")
    else:
        indices = None

    try:
        free_memory, _ = cp.cuda.runtime.memGetInfo()
        bytes_per_iteration = max(1, n_observations * X.shape[1] * 8 * 3)
        memory_batch = max(1, int((free_memory * 0.35) // bytes_per_iteration))
        batch_size = max(1, min(batch_size, memory_batch))
    except cp.cuda.runtime.CUDARuntimeError:
        batch_size = 1

    rng = cp.random.RandomState(random_seed)
    coefficients = []
    start = 0
    while start < iterations:
        count = min(batch_size, iterations - start)
        if indices is None:
            batch_indices = rng.randint(0, n_observations, size=(count, n_observations))
        else:
            batch_indices = cp.asarray(indices[start:start + count])
        try:
            sampled_X = X[batch_indices]
            sampled_y = y[batch_indices]
            beta = cp.matmul(cp.linalg.pinv(sampled_X), sampled_y[..., None])[..., 0]
        except cp.cuda.memory.OutOfMemoryError:
            del sampled_X, sampled_y, batch_indices
            cp.get_default_memory_pool().free_all_blocks()
            if count == 1:
                raise
            batch_size = max(1, count // 2)
            continue
        coefficients.append(beta[:, main_index])
        start += count

    samples = cp.concatenate(coefficients)
    samples_cpu = cp.asnumpy(samples)
    if not np.isfinite(samples_cpu).all():
        raise ValueError("GPU bootstrap generated non-finite coefficients.")
    return BootstrapMetrics(
        mean=float(np.mean(samples_cpu)),
        standard_error=float(np.std(samples_cpu, ddof=1)),
        ci_95=[float(np.percentile(samples_cpu, 2.5)), float(np.percentile(samples_cpu, 97.5))],
        samples=samples_cpu.tolist(),
    )


def run_gpu_analysis(request_data: GPUAnalysisRequest) -> GPUAnalysisOutput:
    started = perf_counter()
    raw_csv = decode_matrix_payload(request_data)
    frame = pd.read_csv(BytesIO(raw_csv))
    required = {
        request_data.outcome_column,
        request_data.main_predictor_column,
        *request_data.bootstrap_predictor_columns,
        *(column for columns in request_data.model_predictor_columns for column in columns),
    }
    if not required.issubset(frame.columns):
        raise ValueError("GPU matrix does not contain every requested column.")
    frame = frame.loc[:, sorted(required)].apply(pd.to_numeric, errors="raise")
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("GPU matrix must contain finite numeric data.")

    y = cp.asarray(frame[request_data.outcome_column].to_numpy(dtype=np.float64))
    models = []
    for predictor_columns in request_data.model_predictor_columns:
        predictors = cp.asarray(frame[predictor_columns].to_numpy(dtype=np.float64))
        X = cp.column_stack([cp.ones(len(frame), dtype=cp.float64), predictors])
        main_index = predictor_columns.index(request_data.main_predictor_column) + 1
        models.append(_fit_ols_gpu(X, y, main_index))

    bootstrap_columns = request_data.bootstrap_predictor_columns
    bootstrap_X = cp.asarray(frame[bootstrap_columns].to_numpy(dtype=np.float64))
    bootstrap_X = cp.column_stack([cp.ones(len(frame), dtype=cp.float64), bootstrap_X])
    bootstrap_main_index = bootstrap_columns.index(request_data.main_predictor_column) + 1
    bootstrap = _bootstrap_gpu(
        bootstrap_X,
        y,
        bootstrap_main_index,
        request_data.bootstrap_iterations,
        request_data.random_seed,
        request_data.bootstrap_indices,
    )
    cp.cuda.Stream.null.synchronize()
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    gpu_name = properties["name"].decode() if isinstance(properties["name"], bytes) else str(properties["name"])
    return GPUAnalysisOutput(
        models=models,
        bootstrap=bootstrap,
        runtime_seconds=perf_counter() - started,
        gpu_name=gpu_name,
    )
