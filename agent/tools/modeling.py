from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
# RandomForest is a versatile and widely-used machine learning algorithm 
# suitable for both classification and regression tasks.

def infer_task(y: pd.Series) -> str:
    return "classification" if y.nunique()<=10 and set(y.unique()).issubset({0,1}) else "regression"

def run_model(csv_path="data/sample.csv", target="churn", test_size=0.2, random_state=42, want_shap=True, run_dir:Path|None=None) -> Dict:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        target = df.columns[-1]
    y = df[target]
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)
    task = infer_task(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    shap_path = None
    if task=="classification":
        model = RandomForestClassifier(n_estimators=300, random_state=random_state)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = [("accuracy", round(float(accuracy_score(y_test, pred)),4)),
                   ("f1", round(float(f1_score(y_test, pred)),4))]
        if want_shap:
            try:
                import shap, matplotlib.pyplot as plt
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                plt.figure()
                shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X_test, show=False)
                shap_path = str((run_dir/"shap_summary.png") if run_dir else Path("data/runs/shap_summary.png"))
                plt.tight_layout()
                plt.savefig(shap_path, dpi=160, bbox_inches="tight")
                plt.close()
            except Exception as e:
                shap_path = None
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=random_state)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = [("MAE", round(float(mean_absolute_error(y_test, pred)),4)),
                   ("R2", round(float(r2_score(y_test, pred)),4))]
    return {
        "name": f"RandomForest ({task})",
        "validation": f"holdout {int((1-test_size)*100)}/{int(test_size*100)} split",
        "metrics": metrics,
        "feature_count": X.shape[1],
        "shap_path": shap_path
    }
