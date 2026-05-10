"""
SHAP analysis on the tuned Extra Trees model.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
DATA = _HERE.parent / "data"
OUT  = _HERE.parent / "output"
OUT.mkdir(exist_ok=True)

TARGET = "C. wuellerstorfi13C"
SPLIT_SEED = 42
TEST_SIZE  = 0.2

ET_PARAMS = dict(
    n_estimators=200, max_depth=15, min_samples_split=10,
    min_samples_leaf=2, random_state=SPLIT_SEED, n_jobs=-1,
)

SELECTED_RATIOS = ["Sr/Ca", "Ba/Ti", "Mn/Al", "Mn/Fe", "Rb/T",
                   "Ca/T", "Fe/T", "Ba/Al", "Zr/T", "Ti/T"]
RD = {
    "Sr/Ca": ("Sr_Area", "Ca_Area"), "Ba/Ti": ("Ba_Area", "Ti_Area"),
    "Mn/Al": ("Mn_Area", "Al_Area"), "Mn/Fe": ("Mn_Area", "Fe_Area"),
    "Rb/T":  ("Rb_Area", "T_Area"),  "Ca/T":  ("Ca_Area", "T_Area"),
    "Fe/T":  ("Fe_Area", "T_Area"),  "Ba/Al": ("Ba_Area", "Al_Area"),
    "Zr/T":  ("Zr_Area", "T_Area"),  "Ti/T":  ("Ti_Area", "T_Area"),
}


def compute_ratios(df, eps=1e-10):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return pd.DataFrame({r: df[RD[r][0]] / (df[RD[r][1]] + eps)
                         for r in SELECTED_RATIOS})


def main():
    df = pd.read_csv(DATA / "data_clean.csv")
    df.columns = df.columns.str.strip()
    y = df[TARGET]
    X = compute_ratios(df)
    keep = X.notna().all(axis=1) & y.notna()
    X, y = X[keep], y[keep]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)

    model = ExtraTreesRegressor(**ET_PARAMS).fit(X_tr_s, y_tr)

    explainer = shap.TreeExplainer(model)
    sh_tr = explainer.shap_values(X_tr_s)
    sh_te = explainer.shap_values(X_te_s)

    X_all  = pd.concat([X_tr_s, X_te_s], axis=0)
    sh_all = np.vstack([sh_tr, sh_te])

    imp = (pd.DataFrame({"Feature": X_tr_s.columns,
                         "Mean_SHAP": np.abs(sh_all).mean(axis=0)})
             .sort_values("Mean_SHAP", ascending=False))
    imp.to_csv(OUT / "shap_importance.csv", index=False)
    # print(imp.to_string(index=False))

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(sh_all, X_all, plot_type="bar",
                      show=False, max_display=len(SELECTED_RATIOS))
    plt.title("SHAP Feature Importance", fontweight="bold", pad=10)
    plt.xlabel("Mean |SHAP value|", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "shap_importance_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "shap_importance_bar.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(sh_all, X_all, plot_type="dot",
                      show=False, max_display=len(SELECTED_RATIOS))
    plt.title("SHAP Summary Plot", fontweight="bold", pad=10)
    plt.xlabel("SHAP value (impact on model output)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT / "shap_summary_beeswarm.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
