from decimal import Decimal
import logging
from types import SimpleNamespace

import pytest

from app import app
from regressionlab.services.export_report import latex_summary
from regressionlab.services.llm_summary import (
    CAUSAL_CAVEAT,
    AnalysisSummary,
    GeneratedSummary,
    LLMSummaryError,
    LLMUsage,
    build_fallback_summary,
    build_summary_facts,
    calculate_llm_cost,
    create_openai_client,
    generate_llm_summary,
    render_analysis_summary_text,
    serialize_summary_facts,
)


PRICING = {
    "input_per_million": Decimal("0.20"),
    "cached_input_per_million": Decimal("0.02"),
    "output_per_million": Decimal("1.25"),
}


def model_result(
    coefficient,
    *,
    model_name="Model 1",
    controls=None,
    p_value=0.01,
    ci_95=None,
):
    return {
        "model_name": model_name,
        "controls": controls or [],
        "coefficient": coefficient,
        "standard_error": 0.1,
        "p_value": p_value,
        "ci_95": ci_95 or [coefficient - 0.2, coefficient + 0.2],
        "n_observations": 100,
    }


def summary_facts(
    *,
    baseline=2.0,
    final=1.5,
    bootstrap_ci=(1.1, 1.9),
):
    return build_summary_facts(
        research_question="Does education predict wages?",
        dependent_variable="wage",
        main_independent_variable="education",
        controls=["experience"],
        model_results=[
            model_result(baseline),
            model_result(
                final,
                model_name="Model 2",
                controls=["experience"],
            ),
        ],
        bootstrap_results={
            "mean": final,
            "standard_error": 0.2,
            "ci_95": list(bootstrap_ci),
        },
        bootstrap_iterations=500,
        compute_mode="CPU",
        runtime_seconds=1.25,
    )


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.responses = FakeResponses(response=response, error=error)


class FakeProviderError(Exception):
    status_code = 429
    request_id = "req_provider_error"


def parsed_response(*, with_usage=True):
    usage = None
    if with_usage:
        usage = SimpleNamespace(
            input_tokens=700,
            input_tokens_details=SimpleNamespace(cached_tokens=200),
            output_tokens=250,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            total_tokens=950,
        )

    return SimpleNamespace(
        output_parsed=GeneratedSummary(
            plain_english_answer="Education has a positive association with wage.",
            main_finding="The final coefficient is 1.5.",
            robustness="The estimate decreases after adding experience.",
            bootstrap_uncertainty="The bootstrap interval excludes zero.",
            suggested_next_checks=[
                "Check robust standard errors.",
                "Inspect influential observations.",
            ],
        ),
        usage=usage,
        model="gpt-5.4-nano-2026-03-17",
        _request_id="req_test_123",
    )


def test_build_summary_facts_captures_attenuation_and_runtime():
    facts = summary_facts(baseline=2.0, final=1.5)

    assert facts.coefficient_change == pytest.approx(-0.5)
    assert facts.relative_coefficient_change == pytest.approx(-0.25)
    assert facts.coefficient_sign_changed is False
    assert facts.bootstrap_interval_includes_zero is False
    assert facts.observation_count == 100
    assert facts.compute_mode == "CPU"
    assert facts.runtime_seconds == pytest.approx(1.25)
    assert len(facts.model_progression) == 2


def test_build_summary_facts_flags_sign_reversal_and_zero_crossing():
    facts = summary_facts(
        baseline=0.8,
        final=-0.2,
        bootstrap_ci=(-0.6, 0.3),
    )

    assert facts.coefficient_sign_changed is True
    assert facts.bootstrap_interval_includes_zero is True
    assert (
        "The main coefficient changes sign across model specifications."
        in facts.diagnostics_warnings
    )
    assert (
        "The 95% bootstrap interval includes zero."
        in facts.diagnostics_warnings
    )


def test_build_summary_facts_preserves_negative_effect_direction():
    facts = summary_facts(
        baseline=-2.0,
        final=-1.5,
        bootstrap_ci=(-1.9, -1.1),
    )

    assert facts.baseline_coefficient == pytest.approx(-2.0)
    assert facts.final_coefficient == pytest.approx(-1.5)
    assert facts.coefficient_change == pytest.approx(0.5)
    assert facts.coefficient_sign_changed is False
    assert facts.bootstrap_interval_includes_zero is False


def test_build_summary_facts_handles_zero_baseline_and_missing_metrics():
    results = [
        model_result(0.0),
        model_result(
            None,
            model_name="Model 2",
            controls=["experience"],
            p_value=None,
            ci_95=[None, None],
        ),
    ]
    facts = build_summary_facts(
        research_question="Question",
        dependent_variable="outcome",
        main_independent_variable="predictor",
        controls=["experience"],
        model_results=results,
        bootstrap_results={
            "mean": None,
            "standard_error": None,
            "ci_95": [None, None],
        },
        bootstrap_iterations=100,
    )

    assert facts.coefficient_change is None
    assert facts.relative_coefficient_change is None
    assert facts.bootstrap_interval_includes_zero is None
    assert "One or more summary metrics are unavailable." in facts.diagnostics_warnings
    assert "NaN" not in serialize_summary_facts(facts)


def test_missing_api_key_disables_client_without_raising():
    assert create_openai_client(None, timeout_seconds=12) is None


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            LLMUsage(
                input_tokens=700,
                cached_input_tokens=0,
                output_tokens=250,
                reasoning_tokens=0,
                total_tokens=950,
            ),
            Decimal("0.0004525"),
        ),
        (
            LLMUsage(
                input_tokens=700,
                cached_input_tokens=200,
                output_tokens=250,
                reasoning_tokens=0,
                total_tokens=950,
            ),
            Decimal("0.0004165"),
        ),
        (
            LLMUsage(
                input_tokens=700,
                cached_input_tokens=700,
                output_tokens=250,
                reasoning_tokens=0,
                total_tokens=950,
            ),
            Decimal("0.0003265"),
        ),
        (None, None),
    ],
)
def test_calculate_llm_cost(usage, expected):
    assert calculate_llm_cost(usage, PRICING) == expected


def test_generate_summary_uses_structured_single_call_and_safe_logging(caplog):
    facts = summary_facts()
    facts.research_question = "SECRET_RESEARCH_PAYLOAD"
    client = FakeClient(response=parsed_response())
    logger = logging.getLogger("test.llm.success")

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = generate_llm_summary(
            facts,
            client=client,
            model="gpt-5.4-nano-2026-03-17",
            pricing=PRICING,
            max_output_tokens=500,
            logger=logger,
        )

    assert result.generation_status == "generated"
    assert result.causal_caveat == CAUSAL_CAVEAT
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["model"] == "gpt-5.4-nano-2026-03-17"
    assert request["text_format"] is GeneratedSummary
    assert request["text"] == {"verbosity": "low"}
    assert request["reasoning"] == {"effort": "none"}
    assert "verbosity" not in request
    assert request["max_output_tokens"] == 500
    assert request["store"] is False
    assert "SECRET_RESEARCH_PAYLOAD" in request["input"]

    record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "llm_summary.completed"
    )
    assert record.request_id == "req_test_123"
    assert record.input_tokens == 700
    assert record.cached_input_tokens == 200
    assert record.output_tokens == 250
    assert record.estimated_cost_usd == "0.0004165"
    assert "SECRET_RESEARCH_PAYLOAD" not in caplog.text


def test_generate_summary_succeeds_when_usage_is_missing(caplog):
    client = FakeClient(response=parsed_response(with_usage=False))
    logger = logging.getLogger("test.llm.no_usage")

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = generate_llm_summary(
            summary_facts(),
            client=client,
            model="gpt-5.4-nano-2026-03-17",
            pricing=PRICING,
            logger=logger,
        )

    assert result.generation_status == "generated"
    record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "llm_summary.completed"
    )
    assert record.estimated_cost_usd is None


@pytest.mark.parametrize(
    ("client", "expected_error_type"),
    [
        pytest.param(None, "LLMSummaryError", id="missing-api-key"),
        pytest.param(
            FakeClient(error=TimeoutError("timed out")),
            "TimeoutError",
            id="timeout",
        ),
        pytest.param(
            FakeClient(error=FakeProviderError("rate limited")),
            "FakeProviderError",
            id="provider-error",
        ),
        pytest.param(
            FakeClient(error=ValueError("malformed structured output")),
            "ValueError",
            id="malformed-output",
        ),
        pytest.param(
            FakeClient(
                response=SimpleNamespace(
                    output_parsed=None,
                    usage=None,
                    model="gpt-5.4-nano-2026-03-17",
                    _request_id="req_empty",
                )
            ),
            "LLMSummaryError",
            id="refusal-or-empty-output",
        ),
    ],
)
def test_generate_summary_wraps_failures_and_logs_safely(
    client,
    expected_error_type,
    caplog,
):
    logger = logging.getLogger(f"test.llm.failure.{expected_error_type}")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        with pytest.raises(LLMSummaryError):
            generate_llm_summary(
                summary_facts(),
                client=client,
                model="gpt-5.4-nano-2026-03-17",
                pricing=PRICING,
                logger=logger,
            )

    record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "llm_summary.failed"
    )
    assert record.error_type == expected_error_type
    assert "Does education predict wages?" not in caplog.text

    if expected_error_type == "FakeProviderError":
        assert record.status_code == 429
        assert record.request_id == "req_provider_error"


def test_fallback_summary_renders_for_plain_text_and_latex():
    fallback = build_fallback_summary(
        summary_facts(bootstrap_ci=(-0.2, 1.9))
    )

    text = render_analysis_summary_text(fallback)
    latex = latex_summary(fallback.model_dump(mode="json"))

    assert fallback.generation_status == "fallback"
    assert "Plain-English answer:" in text
    assert "Suggested next checks:" in text
    assert CAUSAL_CAVEAT in text
    assert "Plain-English answer:" in latex
    assert CAUSAL_CAVEAT in latex


def test_analyze_renders_structured_fallback_without_openai(
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
            "bootstrap_iterations": "2",
        },
    )

    assert response.status_code == 200
    assert b"Deterministic Research Summary" in response.data
    assert b"Log in to access LLM features" in response.data
    assert b"Plain-English answer" in response.data
    assert b"Suggested next checks" in response.data
    assert CAUSAL_CAVEAT.encode() in response.data
