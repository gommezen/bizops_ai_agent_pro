"""
Data quality profiling utilities.

This module provides simple checks on tabular datasets,
including missing values, duplicates, outlier estimates,
and basic column statistics.
"""

import numpy as np
import pandas as pd


def simple_outlier_count(df: pd.DataFrame) -> int:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return 0
    z = (num - num.mean()) / num.std(ddof=0)
    return int((np.abs(z) > 3).sum().sum())


def run_dq(csv_path="data/sample.csv"):
    df = pd.read_csv(csv_path)
    rows, cols = df.shape
    missing_pct = float(df.isna().mean().median() * 100)
    duplicates = int(df.duplicated().sum())
    outliers = simple_outlier_count(df)
    nunique = df.nunique().to_dict()
    return {
        "rows": int(rows),
        "cols": int(cols),
        "missing_pct": round(missing_pct, 2),
        "duplicates": duplicates,
        "outliers": outliers,
        "nunique": {k: int(v) for k, v in nunique.items()},
        "columns": df.columns.tolist(),
    }
