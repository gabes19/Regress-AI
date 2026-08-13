#This service handles model fitting
import numpy as np
import statsmodels.api as sm
from .data_processing import PreparedAnalysisData

def clean_metric(value):
    '''Return JSON/template-friendly floats for model metrics.'''
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(metric):
        return None

    return metric


def format_p_value(value, digits=4):
    """Format p-values without presenting rounded underflow as exact zero."""
    metric = clean_metric(value)
    if metric is None:
        return "n/a"

    threshold = 10 ** -digits
    if 0 <= metric < threshold:
        return f"<{threshold:.{digits}f}"

    return f"{metric:.{digits}f}"

def fit_models(
    data: PreparedAnalysisData,
    dependent_variable: str,
    main_independent_variable: str,
    controls: list[str],
):
    """Fit the baseline and control-progression models."""

    y = data.y
    model_results = []

    for index in range(len(controls) + 1):
        current_controls = controls[:index]

        model_columns = list(
            data.term_map[main_independent_variable]
        )

        
        for control in current_controls:
            model_columns.extend(data.term_map[control])

        X = data.X[model_columns]
        X = sm.add_constant(X, has_constant="add")

        model = sm.OLS(y, X).fit()

        coefficient_interval = model.conf_int().loc[
            main_independent_variable
        ]

        formula_terms = [
            main_independent_variable,
            *current_controls,
        ]

        model_results.append({
            "model_name": f"Model {index + 1}",
            "formula": (
                f"{dependent_variable} ~ "
                + " + ".join(formula_terms)
            ),
            "controls": current_controls,
            "coefficient": clean_metric(
                model.params[main_independent_variable]
            ),
            "standard_error": clean_metric(
                model.bse[main_independent_variable]
            ),
            "t_value": clean_metric(
                model.tvalues[main_independent_variable]
            ),
            "p_value": clean_metric(
                model.pvalues[main_independent_variable]
            ),
            "ci_95": [
                clean_metric(coefficient_interval.iloc[0]),
                clean_metric(coefficient_interval.iloc[1]),
            ],
            "r_squared": clean_metric(model.rsquared),
            "adjusted_r_squared": clean_metric(
                model.rsquared_adj
            ),
            "rmse": clean_metric(np.sqrt(model.mse_resid)),
            "f_statistic": clean_metric(model.fvalue),
            "f_p_value": clean_metric(model.f_pvalue),
            "n_observations": int(model.nobs),
            "df_residual": clean_metric(model.df_resid),
            "condition_number": clean_metric(
                model.condition_number
            ),
        })

    return model_results
