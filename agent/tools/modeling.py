import inspect

from sklearn.preprocessing import OneHotEncoder

# Compat: sklearn >=1.2 uses `sparse_output`; older uses `sparse`
if "sparse_output" in inspect.signature(OneHotEncoder).parameters:

    def OHE(**kw):  # dense output
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore", **kw)

else:

    def OHE(**kw):  # dense output for older versions
        return OneHotEncoder(sparse=False, handle_unknown="ignore", **kw)


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _split_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def infer_task(y: pd.Series) -> str:
    # Binary or small integer label sets => classification, else regression
    vals = pd.unique(y.dropna())
    if len(vals) <= 10 and set(pd.Series(vals).astype(str)).issubset({"0", "1"}):
        return "classification"
    if set(vals).issubset({0, 1}):
        return "classification"
    return "regression"


def run_model(
    csv_path: str = "data/sample.csv",
    target: str = "churn",
    test_size: float = 0.2,
    random_state: int = 42,
    want_shap: bool = False,
    run_dir: Path | None = None,
) -> dict:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        target = df.columns[-1]

    # Basic cleanup
    df = df.dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])

    # Column typing
    num_cols, cat_cols = _split_columns(df, target)

    # Preprocess: one-hot for categoricals (sparse), passthrough numerics
    pre = ColumnTransformer(
        transformers=[
            ("cat", OHE(), cat_cols),
        ]
    )

    task = infer_task(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task == "classification" else None,
    )

    if task == "classification":
        model = HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_depth=None,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
        )
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = [
            ("accuracy", round(float(accuracy_score(y_test, pred)), 4)),
            ("f1", round(float(f1_score(y_test, pred, zero_division=0)), 4)),
        ]
        # Optional SHAP (sampled) — can be slower; keep off by default
        shap_path = None
        if want_shap:
            try:
                import matplotlib.pyplot as plt
                import shap

                shap_n = min(400, X_test.shape[0])
                X_shap = X_test.sample(shap_n, random_state=42)
                # SHAP on gradient boosting needs background; use training sample
                explainer = shap.Explainer(pipe.named_steps["clf"])
                shap_values = explainer(pre.transform(X_shap))
                plt.figure()
                shap.summary_plot(shap_values, pre.transform(X_shap), show=False)
                if run_dir is None:
                    run_dir = Path("data/runs")
                run_dir.mkdir(parents=True, exist_ok=True)
                shap_path = str(run_dir / "shap_summary.png")
                plt.tight_layout()
                plt.savefig(shap_path, dpi=140, bbox_inches="tight")
                plt.close()
            except Exception:
                shap_path = None

        name = "HistGradientBoosting (classification)"

    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.07,
            max_depth=None,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
        )
        pipe = Pipeline([("pre", pre), ("reg", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = [
            ("MAE", round(float(mean_absolute_error(y_test, pred)), 4)),
            ("R2", round(float(r2_score(y_test, pred)), 4)),
        ]
        shap_path = None
        name = "HistGradientBoosting (regression)"

    return {
        "name": name,
        "validation": f"holdout {int((1 - test_size) * 100)}/{int(test_size * 100)} split",
        "metrics": metrics,
        "feature_count": len(num_cols) + len(cat_cols),
        "shap_path": shap_path or "",
    }
