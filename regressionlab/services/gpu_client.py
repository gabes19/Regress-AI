"""Narrow RunPod Serverless client with safe operational logging."""

from __future__ import annotations

from decimal import Decimal
import json
import logging
import math
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .gpu_contract import GPUAnalysisOutput, GPUAnalysisRequest, RunPodResult


class GPUServiceError(Exception):
    """Raised when a GPU request cannot produce a validated result."""


def calculate_gpu_cost(execution_time_ms, price_per_second=Decimal("0.0002")):
    if execution_time_ms is None:
        return None
    billed_seconds = math.ceil(max(0, int(execution_time_ms)) / 1000)
    return Decimal(billed_seconds) * Decimal(price_per_second)


class RunPodClient:
    def __init__(
        self,
        endpoint_id,
        api_key,
        wait_milliseconds=90_000,
        timeout_seconds=100,
        execution_timeout_seconds=60,
        price_per_second=Decimal("0.0002"),
        logger=None,
    ):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.wait_milliseconds = wait_milliseconds
        self.timeout_seconds = timeout_seconds
        self.execution_timeout_seconds = execution_timeout_seconds
        self.price_per_second = Decimal(price_per_second)
        self.logger = logger or logging.getLogger(__name__)

    @property
    def configured(self):
        return bool(self.endpoint_id and self.api_key)

    def run(self, analysis_request: GPUAnalysisRequest) -> RunPodResult:
        if not self.configured:
            raise GPUServiceError("Cloud GPU is not configured.")

        url = (
            f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
            f"?wait={self.wait_milliseconds}"
        )
        payload = json.dumps({
            "input": analysis_request.model_dump(mode="json"),
            "policy": {
                "executionTimeout": self.execution_timeout_seconds * 1000,
                "ttl": 120_000,
            },
        }, allow_nan=False).encode("utf-8")
        http_request = Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        started = perf_counter()
        request_id = None
        try:
            with urlopen(http_request, timeout=self.timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            request_id = envelope.get("id")
            if envelope.get("status") != "COMPLETED" or "output" not in envelope:
                raise GPUServiceError(
                    f"RunPod job did not complete (status={envelope.get('status', 'unknown')})."
                )
            output = GPUAnalysisOutput.model_validate(envelope["output"])
            elapsed = perf_counter() - started
            result = RunPodResult(
                job_id=request_id,
                delay_time_ms=envelope.get("delayTime"),
                execution_time_ms=envelope.get("executionTime"),
                end_to_end_seconds=elapsed,
                output=output,
            )
            cost = calculate_gpu_cost(result.execution_time_ms, self.price_per_second)
            self.logger.info(
                "GPU analysis succeeded provider=runpod request_id=%s latency_ms=%.2f execution_ms=%s estimated_cost_usd=%s",
                request_id,
                elapsed * 1000,
                result.execution_time_ms,
                cost if cost is not None else "unavailable",
            )
            return result
        except GPUServiceError:
            raise
        except HTTPError as error:
            self._log_failure(started, type(error).__name__, error.code, request_id)
            raise GPUServiceError("Cloud GPU rejected the analysis request.") from error
        except (URLError, TimeoutError) as error:
            self._log_failure(started, type(error).__name__, None, request_id)
            raise GPUServiceError("Cloud GPU was unavailable or timed out.") from error
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as error:
            self._log_failure(started, type(error).__name__, None, request_id)
            raise GPUServiceError("Cloud GPU returned an invalid response.") from error

    def _log_failure(self, started, error_type, status_code, request_id):
        self.logger.warning(
            "GPU analysis failed provider=runpod request_id=%s latency_ms=%.2f error_type=%s status_code=%s",
            request_id,
            (perf_counter() - started) * 1000,
            error_type,
            status_code,
        )
