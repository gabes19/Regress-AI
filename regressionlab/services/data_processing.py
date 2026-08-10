#Handles parsing and data processing for input
import pandas as pd
import numpy as np
from dataclasses import dataclass

#Define maximum amount of categories for one-hot encoding
MAX_AUTO_CATEGORIES = 25


class CSVValidationError(ValueError):
    """Raised when a CSV violates an application intake limit."""


def validate_csv_shape(csv_path, max_rows: int, max_columns: int) -> None:
    """Reject oversized CSV shapes before loading the full dataset in memory."""
    header = pd.read_csv(csv_path, nrows=0)
    column_count = len(header.columns)
    if column_count == 0:
        raise CSVValidationError("The CSV must contain at least one column.")
    if column_count > max_columns:
        raise CSVValidationError(
            f"The CSV has {column_count:,} columns; the maximum is "
            f"{max_columns:,}."
        )

    row_count = 0
    for chunk in pd.read_csv(csv_path, chunksize=10_000):
        row_count += len(chunk)
        if row_count > max_rows:
            raise CSVValidationError(
                f"The CSV has more than {max_rows:,} rows, which is the "
                "current maximum."
            )

    if row_count == 0:
        raise CSVValidationError("The CSV must contain at least one data row.")

@dataclass
class PreparedAnalysisData:
    y: pd.Series # cleaned dependent variable
    X: pd.DataFrame # all numeric and encoded predictors
    term_map: dict[str, list[str]] # mapping from original controls to resulting columns

def prepare_analysis_data(df : pd.DataFrame,
                          dependent_variable : str,
                          main_independent_variable: str,
                          controls : list[str],
) -> PreparedAnalysisData:
    '''Validate and convert raw cols into data for model input'''

    if dependent_variable == main_independent_variable:
        raise ValueError(
            "The dependent and main independent variable must be different."
        )

    if dependent_variable in controls:
        raise ValueError(
            "The dependent variable cannot also be a control."
        )

    if main_independent_variable in controls:
        raise ValueError(
            "The main independent variable cannot also be a control."
            )

    if len(controls) != len(set(controls)):
        raise ValueError("Controls cannot contain duplicate columns.")

    required_columns = [
        dependent_variable,
        main_independent_variable,
        *controls,
    ]

    # Check selections before indexing the DataFrame.
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"Selected columns were not found in the dataset: {missing_text}"
        )

     # Use one complete-case sample for every progression model and bootstrap.
    model_frame = df.loc[:, required_columns].copy()
    model_frame = model_frame.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    model_frame = model_frame.dropna()

    if model_frame.empty:
        raise ValueError(
            "No complete observations remain after removing missing values."
        )

    # The current application reports one coefficient for the main variable,
    # so dependent and main variables must be numeric.
    dependent_series = model_frame[dependent_variable]
    main_series = model_frame[main_independent_variable]

    if (
        is_categorical_series(dependent_series)
        or not pd.api.types.is_numeric_dtype(dependent_series.dtype)
    ):
        raise ValueError(
            f"Dependent variable '{dependent_variable}' must be numeric."
        )

    if (
        is_categorical_series(main_series)
        or not pd.api.types.is_numeric_dtype(main_series.dtype)
    ):
        raise ValueError(
            f"Main independent variable "
            f"'{main_independent_variable}' must be numeric."
        )

    y = pd.to_numeric(
        dependent_series,
        errors="raise",
    ).astype(float)

    main_data = pd.to_numeric(
        main_series,
        errors="raise",
    ).astype(float).to_frame(name=main_independent_variable)

    predictor_parts = [main_data]

    term_map = {
        main_independent_variable: [main_independent_variable],
    }

    # Process each control separately so its encoded columns can be tracked.
    for control in controls:
        control_series = model_frame[control]

        if is_categorical_series(control_series):
            unique_values = int(
                control_series.nunique(dropna=True)
            )

            if unique_values < 2:
                raise ValueError(
                    f"Categorical control '{control}' must have "
                    "at least two categories."
                )

            if unique_values > MAX_AUTO_CATEGORIES:
                raise ValueError(
                    f"Categorical control '{control}' has "
                    f"{unique_values} categories. The maximum supported "
                    f"is {MAX_AUTO_CATEGORIES}."
                )

            # Use the same deterministic reference level reported by
            # parse_columns().
            category_levels = get_category_levels(control_series)

            categorical_series = pd.Series(
                pd.Categorical(
                    control_series.astype(str),
                    categories=category_levels,
                ),
                index=model_frame.index,
                name=control,
            )

            encoded_control = pd.get_dummies(
                categorical_series,
                prefix=control,
                drop_first=True,
                dtype=float,
            )

            predictor_parts.append(encoded_control)
            term_map[control] = encoded_control.columns.tolist()

        elif pd.api.types.is_numeric_dtype(control_series.dtype):
            numeric_control = pd.to_numeric(
                control_series,
                errors="raise",
            ).astype(float).to_frame(name=control)

            predictor_parts.append(numeric_control)
            term_map[control] = [control]

        else:
            raise ValueError(
                f"Control '{control}' has unsupported dtype "
                f"'{control_series.dtype}'."
            )

    X = pd.concat(predictor_parts, axis=1).astype(float)

    duplicate_encoded_columns = (
        X.columns[X.columns.duplicated()].tolist()
    )

    if duplicate_encoded_columns:
        duplicate_text = ", ".join(duplicate_encoded_columns)
        raise ValueError(
            "Encoding produced duplicate predictor names: "
            f"{duplicate_text}"
        )

    return PreparedAnalysisData(
        y=y,
        X=X,
        term_map=term_map,
    )


def is_categorical_series(series: pd.Series) -> bool:
    '''Helper function that checks if a series is categorical
    for later encoding'''
    dtype = series.dtype
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    )

def get_category_levels(series: pd.Series) -> list:
    '''Helper function that returns a list of category labels sorted by frequency desc'''
    counts = series.dropna().astype(str).value_counts()
    return sorted(
        counts.index.tolist(), 
        key= lambda level: (-int(counts[level]), level.casefold()),
    )

def parse_columns(csv_path):
    '''Helper function to parse column metadata for
    user to choose target variables and prepare for LLM'''

    df = pd.read_csv(csv_path)
    column_metadata = []

    for column in df.columns:
        series = df[column]
        is_categorical = is_categorical_series(series)

        category_levels = (
            get_category_levels(series)
            if is_categorical
            else[]
        )

        unique_values = int(series.nunique(dropna=True))
        auto_encodable = (
            is_categorical
            and 2 <= unique_values <= MAX_AUTO_CATEGORIES
        )
        column_metadata.append({
            "name": column,
            "dtype": str(series.dtype),
            "semantic_type": (
                "categorical"
                if is_categorical
                else "numeric"
                if pd.api.types.is_numeric_dtype(series.dtype)
                else "unsupported"
            ),
            "missing_values": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique()),
            "auto_encodable": auto_encodable,
            "reference_level": (
                category_levels[0]
                if auto_encodable
                else None
            ),
            "encoded_column_count":(
                unique_values - 1
                if auto_encodable
                else 0
        ),
        })
    return column_metadata
