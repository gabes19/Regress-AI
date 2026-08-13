from regressionlab.services.export_report import build_latex_document
from regressionlab.services.llm_summary import CAUSAL_CAVEAT
from regressionlab.services.regression import format_p_value


def report_payload():
    return {
        "created_at": "2026-08-13T12:00:00+00:00",
        "research_question": "Does education predict wages?",
        "dependent_variable": "wage",
        "main_independent_variable": "education",
        "controls": ["experience"],
        "bootstrap_iterations": 500,
        "models": [
            {
                "model_name": "Model 1",
                "formula": "wage ~ education",
                "coefficient": 2.3,
                "standard_error": 0.1,
                "t_value": 23.0,
                "p_value": 0.000001,
                "ci_95": [2.185, 2.585],
                "r_squared": 0.8,
                "adjusted_r_squared": 0.79,
                "rmse": 1.1,
                "f_statistic": 529.0,
                "f_p_value": 0.0,
                "n_observations": 120,
            }
        ],
        "baseline_coefficient": 2.3,
        "final_coefficient": 2.3,
        "coefficient_change": 0.0,
        "coefficient_chart": [
            {"model_name": "Model 1", "coefficient": 2.3}
        ],
        "bootstrap_results": {
            "mean": 2.4,
            "standard_error": 0.1,
            "ci_95": [2.201, 2.591],
            "samples": [2.2, 2.4, 2.6],
        },
        "llm_summary": {
            "plain_english_answer": "Education is positively associated with wages.",
            "causal_caveat": CAUSAL_CAVEAT,
        },
        "compute_mode": "CPU",
        "runtime_seconds": 0.64,
    }


def test_latex_uses_bootstrap_interval_after_rendering_model_rows():
    latex = build_latex_document(report_payload())

    assert r"\textbf{95\% interval:} 2.201 to 2.591" in latex
    assert "2.185 to 2.585" in latex


def test_latex_includes_causal_caveat_exactly_once():
    latex = build_latex_document(report_payload())

    assert latex.count(CAUSAL_CAVEAT) == 1


def test_latex_formats_tiny_model_p_values_as_thresholds():
    latex = build_latex_document(report_payload())

    assert latex.count("<0.0001") == 2
    assert "0.0000" not in latex


def test_p_value_formatter_preserves_regular_and_missing_values():
    assert format_p_value(0.000001, 4) == "<0.0001"
    assert format_p_value(0.0001, 4) == "0.0001"
    assert format_p_value(0.01234, 4) == "0.0123"
    assert format_p_value(None, 4) == "n/a"
