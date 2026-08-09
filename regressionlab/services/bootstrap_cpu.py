# Handles non-intensive bootstrapping jobs.
import numpy as np
import statsmodels.api as sm

from .data_processing import PreparedAnalysisData


def bootstrap_coefficient(
    data: PreparedAnalysisData,
    main_independent_variable: str,
    iterations: int,
    random_seed: int | None = None,
    bootstrap_indices=None,
):
    """Bootstrap the main coefficient from prepared model data."""

    if iterations < 2:
        raise ValueError("Bootstrap iterations must be at least 2.")

    observation_count = len(data.y)

    if observation_count < 2:
        raise ValueError(
            "At least two complete observations are required."
        )

    if main_independent_variable not in data.X.columns:
        raise ValueError(
            f"Main independent variable '{main_independent_variable}' "
            "is missing from the prepared predictors."
        )

    random_generator = np.random.default_rng(random_seed)
    coefficients = np.empty(iterations, dtype=float)

    if bootstrap_indices is not None:
        bootstrap_indices = np.asarray(bootstrap_indices, dtype=np.int64)
        if bootstrap_indices.shape != (iterations, observation_count):
            raise ValueError("Explicit bootstrap indices have the wrong shape.")
        if bootstrap_indices.min(initial=0) < 0 or bootstrap_indices.max(initial=0) >= observation_count:
            raise ValueError("Explicit bootstrap indices are out of range.")

    for iteration in range(iterations):
        sample_positions = (
            bootstrap_indices[iteration]
            if bootstrap_indices is not None
            else random_generator.integers(
                low=0,
                high=observation_count,
                size=observation_count,
            )
        )

        sample_y = data.y.iloc[sample_positions].reset_index(drop=True)
        sample_X = data.X.iloc[sample_positions].reset_index(drop=True)
        sample_X = sm.add_constant(sample_X, has_constant="add")

        model = sm.OLS(sample_y, sample_X).fit()
        coefficients[iteration] = float(
            model.params[main_independent_variable]
        )

    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Bootstrap generated non-finite coefficients.")

    return {
        "mean": float(np.mean(coefficients)),
        "standard_error": float(np.std(coefficients, ddof=1)),
        "ci_95": [
            float(np.percentile(coefficients, 2.5)),
            float(np.percentile(coefficients, 97.5)),
        ],
        "samples": coefficients.tolist(),
    }
