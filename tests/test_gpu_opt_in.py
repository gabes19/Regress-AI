import os

import numpy as np
import pandas as pd
import pytest

from regressionlab.services.bootstrap_cpu import bootstrap_coefficient
from regressionlab.services.data_processing import prepare_analysis_data
from regressionlab.services.gpu_client import RunPodClient
from regressionlab.services.gpu_contract import build_gpu_request
from regressionlab.services.regression import fit_models


def parity_fixture():
    frame = pd.DataFrame({
        "y": [2.0, 3.0, 5.0, 7.0, 8.0, 11.0, 13.0, 15.0],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "group": ["A", "B", "A", "B", "A", "B", "A", "B"],
    })
    return prepare_analysis_data(frame, "y", "x", ["group"])


@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1")
def test_local_cuda_worker_matches_cpu_with_explicit_indices():
    from gpu_worker.regression_gpu import run_gpu_analysis

    prepared = parity_fixture()
    indices = np.random.default_rng(42).integers(0, 8, size=(25, 8)).tolist()
    cpu_models = fit_models(prepared, "y", "x", ["group"])
    cpu_bootstrap = bootstrap_coefficient(
        prepared, "x", 25, bootstrap_indices=indices
    )
    output = run_gpu_analysis(build_gpu_request(
        prepared, "x", ["group"], 25, bootstrap_indices=indices
    ))
    for cpu_model, gpu_model in zip(cpu_models, output.models):
        assert gpu_model.coefficient == pytest.approx(cpu_model["coefficient"], rel=1e-6)
        assert gpu_model.standard_error == pytest.approx(cpu_model["standard_error"], rel=1e-5)
        assert gpu_model.p_value == pytest.approx(cpu_model["p_value"], rel=1e-5)
    assert output.bootstrap.samples == pytest.approx(cpu_bootstrap["samples"], rel=1e-6)


@pytest.mark.live_runpod
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_RUNPOD_TESTS") != "1",
    reason="set RUN_LIVE_RUNPOD_TESTS=1 to authorize a billable smoke test",
)
def test_live_runpod_smoke():
    endpoint_id = os.environ["RUNPOD_ENDPOINT_ID"]
    api_key = os.environ["RUNPOD_API_KEY"]
    request_data = build_gpu_request(parity_fixture(), "x", ["group"], 2)
    result = RunPodClient(endpoint_id, api_key).run(request_data)
    assert result.output.gpu_name
    assert len(result.output.models) == 2
    assert len(result.output.bootstrap.samples) == 2
