"""
Data validation stage — the quality gate in front of training.

TODO: Complete the three level functions and the orchestrator.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from pipeline.config import (
    MAX_MISSING_FRACTION,
    MAX_POSITIVE_RATE,
    MIN_POSITIVE_RATE,
    MIN_ROWS,
    PAY_AMT_FEATURES,
    RAW_FEATURES,
    TARGET,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset fails a check that must not be ignored."""


# Value domains, from the dataset documentation. (PROVIDED)
DOMAINS: Dict[str, Any] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}
RANGES: Dict[str, tuple] = {
    "LIMIT_BAL": (10_000, 2_000_000),
    "AGE": (18, 100),
    **{c: (-2, 8) for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
}


# =============================================================================
# TODO 1: Implement validate_schema — level 1
# =============================================================================
# "Is the data shaped the way the code expects?"
#
# Requirements:
#   - every column in RAW_FEATURES + [TARGET] must be present
#   - every one of those columns must be a numeric dtype
#   - return a LIST OF STRINGS describing what is wrong; empty list means clean
#
# Return errors rather than raising, so the caller can collect all three levels
# and report them together. A validator that stops at the first problem makes
# you fix issues one deploy at a time.
#
# Hint: pd.api.types.is_numeric_dtype(df[col])

def validate_schema(df: pd.DataFrame) -> List[str]:
    """Level 1 — are the expected columns present, with usable types?"""
    errors: List[str] = []
    expected = RAW_FEATURES + [TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    non_numeric = [
        c for c in expected
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        errors.append(f"non-numeric columns: {non_numeric}")
    return errors


# =============================================================================
# TODO 2: Implement validate_statistics — level 2
# =============================================================================
# "Is the shape of the distribution what training assumes?"
#
# Requirements:
#   - at least MIN_ROWS rows
#   - no column more than MAX_MISSING_FRACTION missing
#   - target positive rate inside [MIN_POSITIVE_RATE, MAX_POSITIVE_RATE]
#
# The target check is the one that earns its keep. If an upstream extract breaks
# and every label comes back 0, training still succeeds — you get a model with
# 100% accuracy that predicts "no default" for everyone. This check catches that
# before anyone celebrates.

def validate_statistics(df: pd.DataFrame) -> List[str]:
    """Level 2 — is the shape of the data what training assumes?"""
    errors: List[str] = []
    if len(df) < MIN_ROWS:
        errors.append(f"too few rows: {len(df)} < {MIN_ROWS}")
    if len(df) > 0:
        for col, frac in df.isna().mean().items():
            if frac > MAX_MISSING_FRACTION:
                errors.append(
                    f"column '{col}' has {frac:.1%} missing values (max {MAX_MISSING_FRACTION:.1%})"
                )
    if TARGET in df.columns and len(df) > 0:
        rate = df[TARGET].mean()
        if not (MIN_POSITIVE_RATE <= rate <= MAX_POSITIVE_RATE):
            errors.append(
                f"target positive rate {rate:.3f} outside [{MIN_POSITIVE_RATE}, {MAX_POSITIVE_RATE}]"
            )
    return errors


# =============================================================================
# TODO 3: Implement validate_semantics — level 3
# =============================================================================
# "Do the values mean what the business says they mean?"
#
# Requirements:
#   - every column in DOMAINS may only contain the allowed values
#   - every column in RANGES must stay inside its [lo, hi]
#   - no PAY_AMT* column may contain a negative number
#
# Level 3 is where domain knowledge lives. A SEX of 7 or an AGE of 400 is
# perfectly valid as an integer and perfectly meaningless as a customer.

def validate_semantics(df: pd.DataFrame) -> List[str]:
    """Level 3 — do the values mean what the business says they mean?"""
    errors: List[str] = []
    for col, allowed in DOMAINS.items():
        if col in df.columns:
            bad = sorted(set(df[col].dropna().unique()) - allowed)
            if bad:
                errors.append(f"{col} has values outside {sorted(allowed)}: {bad[:10]}")
    for col, (lo, hi) in RANGES.items():
        if col in df.columns:
            n_bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_bad:
                errors.append(f"{col} has {n_bad} values outside [{lo}, {hi}]")
    for col in PAY_AMT_FEATURES:
        if col in df.columns:
            n_neg = int((df[col] < 0).sum())
            if n_neg:
                errors.append(f"{col} has {n_neg} negative values")
    return errors


# =============================================================================
# TODO 4: Implement validate_dataset
# =============================================================================
# Requirements:
#   - run all three levels and concatenate their errors
#   - build and return this report:
#         {"passed": bool, "n_rows": int, "n_columns": int,
#          "schema_errors": [...], "statistical_errors": [...],
#          "semantic_errors": [...], "n_errors": int}
#   - log every error at ERROR level
#   - if there are errors and raise_on_error is True, raise DataValidationError
#
# The report is logged to MLflow as an artifact by the training stage, so a
# model trained on data with known problems carries the evidence with it.

def validate_dataset(df: pd.DataFrame, raise_on_error: bool = True) -> Dict[str, Any]:
    """Run all three levels and return a report."""
    schema_errors = validate_schema(df)
    statistical_errors = validate_statistics(df)
    semantic_errors = validate_semantics(df)
    all_errors = schema_errors + statistical_errors + semantic_errors

    report = {
        "passed": not all_errors,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "schema_errors": schema_errors,
        "statistical_errors": statistical_errors,
        "semantic_errors": semantic_errors,
        "n_errors": len(all_errors),
    }
    for err in all_errors:
        logger.error("Validation: %s", err)
    if all_errors and raise_on_error:
        raise DataValidationError(f"{len(all_errors)} validation error(s): {all_errors}")
    return report
