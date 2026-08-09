"""RunPod Serverless entrypoint."""

import runpod
from pydantic import ValidationError

from regressionlab.services.gpu_contract import GPUAnalysisRequest
from gpu_worker.regression_gpu import run_gpu_analysis


def handler(job):
    try:
        request_data = GPUAnalysisRequest.model_validate(job.get("input", {}))
        return run_gpu_analysis(request_data).model_dump(mode="json")
    except (ValidationError, ValueError) as error:
        raise ValueError("Invalid GPU analysis request.") from error


runpod.serverless.start({"handler": handler})
