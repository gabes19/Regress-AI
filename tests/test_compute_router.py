from decimal import Decimal

import pandas as pd
import pytest
import app as app_module

from regressionlab.services.compute_router import (
    ComputeUnavailableError,
    run_analysis_compute,
    should_route_to_gpu,
)
from regressionlab.services.data_processing import prepare_analysis_data
from regressionlab.services.gpu_contract import (
    BootstrapMetrics,
    GPUAnalysisOutput,
    ModelMetrics,
    RunPodResult,
)
from regressionlab.services.gpu_usage import initialize_gpu_database, upsert_google_user
from regressionlab.services.gpu_client import GPUServiceError


def prepared_data():
    frame = pd.DataFrame({
        "y": [2, 4, 5, 8, 9, 12, 13, 16],
        "x": [1, 2, 3, 4, 5, 6, 7, 8],
    })
    return prepare_analysis_data(frame, "y", "x", [])


def config(tmp_path):
    database = tmp_path / "gpu.sqlite"
    initialize_gpu_database(database)
    user = upsert_google_user(database, {
        "sub": "1", "email": "u@example.com", "email_verified": True
    })
    return {
        "RUNPOD_ENABLED": True, "GPU_OPT_IN_ITERATION_THRESHOLD": 2,
        "GPU_MIN_WORK_UNITS": 1, "CPU_FALLBACK_MAX_WORK_UNITS": 1_000_000,
        "GPU_MAX_ENCODED_PAYLOAD_BYTES": 15 * 1024 * 1024,
        "GPU_MAX_DECOMPRESSED_BYTES": 256 * 1024 * 1024,
        "GPU_MAX_BOOTSTRAP_ITERATIONS": 10_000,
        "GPU_USAGE_DATABASE": database, "GPU_RESERVED_COST_USD": Decimal("0.012"),
        "GPU_DAILY_USER_LIMIT": 3, "GPU_MONTHLY_USER_LIMIT": 30,
        "GPU_GLOBAL_DAILY_BUDGET_USD": Decimal("2"),
        "GPU_GLOBAL_MONTHLY_BUDGET_USD": Decimal("25"),
        "GPU_GLOBAL_MAX_IN_FLIGHT": 1, "GPU_PRICE_PER_SECOND_USD": Decimal("0.0002"),
    }, user


def metric():
    return ModelMetrics(
        coefficient=2, standard_error=.1, t_value=20, p_value=.001,
        ci_95=[1.8, 2.2], r_squared=.9, adjusted_r_squared=.88,
        rmse=.2, f_statistic=400, f_p_value=.001, n_observations=8,
        df_residual=6, condition_number=10,
    )


class FakeGPUClient:
    configured = True
    calls = 0

    def run(self, request_data):
        self.calls += 1
        return RunPodResult(
            job_id="job-1", execution_time_ms=500, end_to_end_seconds=.6,
            output=GPUAnalysisOutput(
                models=[metric()],
                bootstrap=BootstrapMetrics(
                    mean=2, standard_error=.1, ci_95=[1.8, 2.2], samples=[1.9, 2.1]
                ),
                runtime_seconds=.2, gpu_name="A4000",
            ),
        )


def test_eligible_analysis_runs_gpu_once_without_cpu_duplication(tmp_path, monkeypatch):
    settings, user = config(tmp_path)
    client = FakeGPUClient()

    def cpu_must_not_run(*args, **kwargs):
        raise AssertionError("CPU path ran during a successful GPU analysis")

    monkeypatch.setattr("regressionlab.services.compute_router._run_cpu", cpu_must_not_run)
    result = run_analysis_compute(
        prepared_data(), "y", "x", [], 2, settings, client, user["id"]
    )
    assert client.calls == 1
    assert result.compute_mode == "GPU"


def test_measured_work_threshold_and_high_iteration_consent_both_apply():
    data = prepared_data()

    assert should_route_to_gpu(data, 2_000, 1, 2_000) is True
    assert should_route_to_gpu(data, 2_001, 1, 2_000) is False
    assert should_route_to_gpu(
        data, 2_001, 1, 2_000, gpu_opt_in=True
    ) is True
    assert should_route_to_gpu(
        data, 2_001, 1_000_000, 2_000, gpu_opt_in=True
    ) is False


def test_qualifying_high_iteration_workload_requires_explicit_gpu_consent(
    tmp_path,
):
    settings, user = config(tmp_path)
    client = FakeGPUClient()

    with pytest.raises(ComputeUnavailableError, match="Enable the cloud GPU"):
        run_analysis_compute(
            prepared_data(), "y", "x", [], 3, settings, client, user["id"]
        )

    result = run_analysis_compute(
        prepared_data(), "y", "x", [], 3, settings, client, user["id"],
        gpu_opt_in=True,
    )
    assert result.compute_mode == "GPU"


def test_unsigned_analysis_stays_on_cpu(tmp_path):
    settings, _ = config(tmp_path)
    settings["CPU_FALLBACK_MAX_WORK_UNITS"] = 1
    client = FakeGPUClient()
    result = run_analysis_compute(prepared_data(), "y", "x", [], 2, settings, client, None)
    assert result.compute_mode == "CPU"
    assert client.calls == 0


def test_large_quota_rejection_does_not_start_unsafe_cpu_fallback(tmp_path):
    settings, user = config(tmp_path)
    settings["CPU_FALLBACK_MAX_WORK_UNITS"] = 1
    first = FakeGPUClient()
    # Leave a reservation in flight by reserving through one successful-looking candidate is
    # unnecessary; a globally busy database is tested via the first client raising before reconciliation.
    from regressionlab.services.gpu_usage import reserve_gpu_run
    reserve_gpu_run(
        settings["GPU_USAGE_DATABASE"], user["id"], 1, 2, Decimal("0.012"),
        3, 30, Decimal("2"), Decimal("25")
    )
    with pytest.raises(ComputeUnavailableError, match="Reduce"):
        run_analysis_compute(prepared_data(), "y", "x", [], 2, settings, first, user["id"])


def test_flask_renders_cpu_fallback_after_provider_failure(
    sample_dataset_client, tmp_path, monkeypatch,
):
    client, dataset = sample_dataset_client
    database = tmp_path / "route-gpu.sqlite"
    initialize_gpu_database(database)
    user = upsert_google_user(database, {
        "sub": "route-user", "email": "route@example.com", "email_verified": True
    })

    class FailingGPU:
        configured = True

        def run(self, request_data):
            raise GPUServiceError("provider failed")

    overrides = {
        "GPU_USAGE_DATABASE": database, "RUNPOD_ENABLED": True,
        "GPU_OPT_IN_ITERATION_THRESHOLD": 2, "GPU_MIN_WORK_UNITS": 1,
        "CPU_FALLBACK_MAX_WORK_UNITS": 1_000_000,
    }
    for key, value in overrides.items():
        monkeypatch.setitem(app_module.app.config, key, value)
    monkeypatch.setitem(app_module.app.extensions, "runpod_client", FailingGPU())
    with client.session_transaction() as session:
        session["user"] = user

    response = client.post("/analyze", data={
        "dataset_id": dataset.dataset_id,
        "research_question": "Does education predict wages?",
        "dependent_variable": "wage",
        "main_independent_variable": "education",
        "bootstrap_iterations": "2",
    })
    assert response.status_code == 200
    assert b"CPU fallback" in response.data
