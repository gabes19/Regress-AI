"""Select exactly one primary compute path for an analysis request."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .bootstrap_cpu import bootstrap_coefficient
from .gpu_client import GPUServiceError
from .gpu_contract import build_gpu_request, merge_gpu_model_metadata
from .gpu_usage import GPUQuotaError, reconcile_gpu_run, reserve_gpu_run
from .regression import fit_models


class ComputeUnavailableError(ValueError):
    """Raised when a large workload cannot safely fall back to synchronous CPU."""


@dataclass(frozen=True)
class AnalysisComputeResult:
    model_results: list[dict]
    bootstrap_results: dict
    compute_mode: str
    runtime_seconds: float
    gpu_name: str | None = None
    gpu_execution_time_ms: int | None = None


def calculate_work_units(data, bootstrap_iterations):
    return len(data.y) * (len(data.X.columns) + 1) * bootstrap_iterations


def should_route_to_gpu(
    data,
    bootstrap_iterations,
    minimum_work_units,
    opt_in_iteration_threshold,
    gpu_opt_in=False,
):
    work_units = calculate_work_units(data, bootstrap_iterations)
    workload_qualifies = work_units >= minimum_work_units
    consent_granted = (
        bootstrap_iterations <= opt_in_iteration_threshold or gpu_opt_in
    )
    return workload_qualifies and consent_granted


def _run_cpu(data, dependent_variable, main_independent_variable, controls, iterations, mode):
    started = perf_counter()
    models = fit_models(
        data=data,
        dependent_variable=dependent_variable,
        main_independent_variable=main_independent_variable,
        controls=controls,
    )
    bootstrap = bootstrap_coefficient(
        data=data,
        main_independent_variable=main_independent_variable,
        iterations=iterations,
    )
    return AnalysisComputeResult(models, bootstrap, mode, perf_counter() - started)


def run_analysis_compute(
    data,
    dependent_variable,
    main_independent_variable,
    controls,
    bootstrap_iterations,
    config,
    gpu_client=None,
    user_id=None,
    logger=None,
    gpu_opt_in=False,
):
    work_units = calculate_work_units(data, bootstrap_iterations)
    consent_required = (
        work_units >= config["GPU_MIN_WORK_UNITS"]
        and bootstrap_iterations > config["GPU_OPT_IN_ITERATION_THRESHOLD"]
    )
    if consent_required and not gpu_opt_in:
        raise ComputeUnavailableError(
            "This workload qualifies for cloud GPU computing. Enable the cloud "
            "GPU option or reduce the bootstrap iterations and try again."
        )
    candidate = should_route_to_gpu(
        data,
        bootstrap_iterations,
        config["GPU_MIN_WORK_UNITS"],
        config["GPU_OPT_IN_ITERATION_THRESHOLD"],
        gpu_opt_in,
    )
    provider_available = bool(
        config.get("RUNPOD_ENABLED")
        and gpu_client is not None
        and gpu_client.configured
    )
    if not candidate or not provider_available:
        return _run_cpu(
            data, dependent_variable, main_independent_variable,
            controls, bootstrap_iterations, "CPU"
        )
    if user_id is None:
        return _fallback_or_raise(
            data, dependent_variable, main_independent_variable, controls,
            bootstrap_iterations, work_units, config,
            "A signed-in account is required for cloud GPU."
        )

    try:
        gpu_request = build_gpu_request(
            data=data,
            main_independent_variable=main_independent_variable,
            controls=controls,
            bootstrap_iterations=bootstrap_iterations,
            max_encoded_bytes=config["GPU_MAX_ENCODED_PAYLOAD_BYTES"],
            max_decompressed_bytes=config["GPU_MAX_DECOMPRESSED_BYTES"],
            max_bootstrap_iterations=config["GPU_MAX_BOOTSTRAP_ITERATIONS"],
        )
    except ValueError as error:
        return _fallback_or_raise(
            data, dependent_variable, main_independent_variable, controls,
            bootstrap_iterations, work_units, config, str(error)
        )

    try:
        reservation = reserve_gpu_run(
            database_path=config["GPU_USAGE_DATABASE"],
            user_id=user_id,
            workload_units=work_units,
            bootstrap_iterations=bootstrap_iterations,
            reserved_cost=config["GPU_RESERVED_COST_USD"],
            daily_user_limit=config["GPU_DAILY_USER_LIMIT"],
            monthly_user_limit=config["GPU_MONTHLY_USER_LIMIT"],
            global_daily_budget=config["GPU_GLOBAL_DAILY_BUDGET_USD"],
            global_monthly_budget=config["GPU_GLOBAL_MONTHLY_BUDGET_USD"],
            global_max_in_flight=config["GPU_GLOBAL_MAX_IN_FLIGHT"],
        )
    except GPUQuotaError as error:
        if logger:
            logger.info("GPU routing skipped reason=%s", type(error).__name__)
        return _fallback_or_raise(
            data, dependent_variable, main_independent_variable, controls,
            bootstrap_iterations, work_units, config, str(error)
        )

    try:
        result = gpu_client.run(gpu_request)
        models = merge_gpu_model_metadata(
            result.output,
            dependent_variable,
            main_independent_variable,
            controls,
        )
        reconcile_gpu_run(
            config["GPU_USAGE_DATABASE"], reservation.run_id, "completed",
            execution_time_ms=result.execution_time_ms,
            provider_job_id=result.job_id,
            price_per_second=config["GPU_PRICE_PER_SECOND_USD"],
        )
        return AnalysisComputeResult(
            model_results=models,
            bootstrap_results=result.output.bootstrap.model_dump(mode="json"),
            compute_mode="GPU",
            runtime_seconds=result.end_to_end_seconds,
            gpu_name=result.output.gpu_name,
            gpu_execution_time_ms=result.execution_time_ms,
        )
    except (GPUServiceError, ValueError) as error:
        reconcile_gpu_run(
            config["GPU_USAGE_DATABASE"], reservation.run_id, "failed",
            price_per_second=config["GPU_PRICE_PER_SECOND_USD"],
        )
        return _fallback_or_raise(
            data, dependent_variable, main_independent_variable, controls,
            bootstrap_iterations, work_units, config, str(error)
        )


def _fallback_or_raise(
    data, dependent_variable, main_independent_variable, controls,
    bootstrap_iterations, work_units, config, reason,
):
    if work_units > config["CPU_FALLBACK_MAX_WORK_UNITS"]:
        raise ComputeUnavailableError(
            "Cloud GPU is unavailable for this workload. Reduce the bootstrap "
            "iterations or try again later."
        )
    return _run_cpu(
        data, dependent_variable, main_independent_variable,
        controls, bootstrap_iterations, "CPU fallback"
    )
