"""
Trains the tuned Extra Trees model on the paired XRF + δ¹³C data
and applies it to the high-resolution XRF scan to produce the
reconstructed δ¹³C time series. A 500-iteration bootstrap is combined
with the test-set RMSE in quadrature for 1σ / 2σ envelopes.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
DATA = _HERE.parent / "data"
OUT  = _HERE.parent / "output"
OUT.mkdir(exist_ok=True)

TARGET = "C. wuellerstorfi13C"
SPLIT_SEED  = 42
TEST_SIZE   = 0.2
N_BOOTSTRAP = 500
Z95 = 1.959963984540054
Z68 = 1.0

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

COLOR_OBS    = "#1f4068"
COLOR_PRED   = "#d23f7a"
COLOR_PRED95 = "#f5d0dc"
COLOR_PRED68 = "#e08aa8"
COLOR_HR     = "#1d6e58"
COLOR_HR95   = "#c0d6ea"
COLOR_HR68   = "#5d94c4"


def compute_ratios(df, eps=1e-10):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return pd.DataFrame({r: df[RD[r][0]] / (df[RD[r][1]] + eps)
                         for r in SELECTED_RATIOS})


def main():
    t0 = time.time()
    df = pd.read_csv(DATA / "data_clean.csv")
    df.columns = df.columns.str.strip()
    y = df[TARGET]
    age = df["Age_Ma"]
    X = compute_ratios(df)
    keep = X.notna().all(axis=1) & y.notna() & age.notna()
    X, y, age = X[keep], y[keep], age[keep]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    age_tr = age.loc[X_tr.index].values
    age_te = age.loc[X_te.index].values

    df_hr = pd.read_csv(DATA / "xrf.csv")
    df_hr.columns = df_hr.columns.str.strip()
    age_hr = df_hr["Age_Ma"]
    X_hr   = compute_ratios(df_hr)
    keep_hr = X_hr.notna().all(axis=1) & age_hr.notna()
    X_hr, age_hr = X_hr[keep_hr], age_hr[keep_hr].values

    n_tr, n_te, n_hr = len(X_tr), len(X_te), len(X_hr)
    # print(f"Train / Test / Highres = {n_tr} / {n_te} / {n_hr}")

    X_tr_np, y_tr_np = X_tr.values, y_tr.values
    X_te_np, y_te_np = X_te.values, y_te.values
    X_hr_np = X_hr.values

    preds_tr = np.empty((N_BOOTSTRAP, n_tr), dtype=np.float32)
    preds_te = np.empty((N_BOOTSTRAP, n_te), dtype=np.float32)
    preds_hr = np.empty((N_BOOTSTRAP, n_hr), dtype=np.float32)

    rng = np.random.default_rng(SPLIT_SEED)
    boot_idx = np.stack([rng.integers(0, n_tr, size=n_tr)
                         for _ in range(N_BOOTSTRAP)])

    for b in range(N_BOOTSTRAP):
        idx = boot_idx[b]
        Xb, yb = X_tr_np[idx], y_tr_np[idx]
        scl = StandardScaler().fit(Xb)
        m = ExtraTreesRegressor(**{**ET_PARAMS, "random_state": b}).fit(
            scl.transform(Xb), yb)
        preds_tr[b] = m.predict(scl.transform(X_tr_np)).astype(np.float32)
        preds_te[b] = m.predict(scl.transform(X_te_np)).astype(np.float32)
        preds_hr[b] = m.predict(scl.transform(X_hr_np)).astype(np.float32)
        if (b + 1) % 50 == 0:
            print(f"  refit {b+1:4d}/{N_BOOTSTRAP}")

    mu_tr = preds_tr.mean(axis=0); sd_tr = preds_tr.std(axis=0, ddof=1)
    mu_te = preds_te.mean(axis=0); sd_te = preds_te.std(axis=0, ddof=1)
    mu_hr = preds_hr.mean(axis=0); sd_hr = preds_hr.std(axis=0, ddof=1)

    rmse_test = float(np.sqrt(np.mean((y_te_np - mu_te) ** 2)))
    test_r2   = float(r2_score(y_te_np, mu_te))
    sig_tot_te = np.sqrt(sd_te ** 2 + rmse_test ** 2)
    sig_tot_hr = np.sqrt(sd_hr ** 2 + rmse_test ** 2)
    sig_tot_tr = np.sqrt(sd_tr ** 2 + rmse_test ** 2)
    # print(f"  Test R² = {test_r2:.3f}   Test RMSE = {rmse_test:.3f} ‰")

    age_paired = np.concatenate([age_tr, age_te])
    obs_paired = np.concatenate([y_tr.values, y_te.values])
    mu_paired  = np.concatenate([mu_tr, mu_te])
    sb_paired  = np.concatenate([sd_tr, sd_te])
    st_paired  = np.concatenate([sig_tot_tr, sig_tot_te])
    o = np.argsort(age_paired)
    age_paired, obs_paired = age_paired[o], obs_paired[o]
    mu_paired, sb_paired, st_paired = mu_paired[o], sb_paired[o], st_paired[o]

    pd.DataFrame({
        "Age_Ma": age_paired, "observed": obs_paired,
        "mean": mu_paired, "sigma": st_paired,
        "lower_2sigma": mu_paired - Z95 * st_paired,
        "lower_1sigma": mu_paired - Z68 * st_paired,
        "upper_1sigma": mu_paired + Z68 * st_paired,
        "upper_2sigma": mu_paired + Z95 * st_paired,
    }).to_csv(OUT / "predictions_paired.csv", index=False)

    o = np.argsort(age_hr)
    age_hr, mu_hr = age_hr[o], mu_hr[o]
    sd_hr, sig_tot_hr = sd_hr[o], sig_tot_hr[o]
    pd.DataFrame({
        "Age_Ma": age_hr, "mean": mu_hr, "sigma": sig_tot_hr,
        "lower_2sigma": mu_hr - Z95 * sig_tot_hr,
        "lower_1sigma": mu_hr - Z68 * sig_tot_hr,
        "upper_1sigma": mu_hr + Z68 * sig_tot_hr,
        "upper_2sigma": mu_hr + Z95 * sig_tot_hr,
    }).to_csv(OUT / "predictions_highres.csv", index=False)

    json.dump({
        "test_R2": test_r2, "test_RMSE_permil": rmse_test,
        "n_bootstrap": N_BOOTSTRAP,
        "elapsed_seconds": time.time() - t0,
    }, open(OUT / "model_metrics.json", "w"), indent=2)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"],
        "font.size": 10, "axes.labelsize": 10,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    age_min = min(age_paired.min(), age_hr.min())
    age_max = max(age_paired.max(), age_hr.max())
    y_min = min(obs_paired.min(),
                (mu_paired - Z95 * st_paired).min(),
                (mu_hr - Z95 * sig_tot_hr).min()) - 0.05
    y_max = max(obs_paired.max(),
                (mu_paired + Z95 * st_paired).max(),
                (mu_hr + Z95 * sig_tot_hr).max()) + 0.05

    fig, axs = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True,
                            gridspec_kw={"hspace": 0.10})

    axs[0].plot(age_paired, obs_paired, color=COLOR_OBS, lw=0.6, alpha=0.6)
    axs[0].scatter(age_paired, obs_paired, s=10, color=COLOR_OBS,
                   edgecolor="white", linewidth=0.3)
    axs[0].set_ylabel("Observed\nδ¹³C (‰)")

    axs[1].errorbar(age_paired, mu_paired,
                    yerr=[Z95 * st_paired, Z95 * st_paired],
                    fmt="none", ecolor=COLOR_PRED95, elinewidth=0.6,
                    alpha=0.85, capsize=0, zorder=2)
    axs[1].errorbar(age_paired, mu_paired,
                    yerr=[Z68 * st_paired, Z68 * st_paired],
                    fmt="none", ecolor=COLOR_PRED68, elinewidth=0.9,
                    alpha=0.85, capsize=0, zorder=3)
    axs[1].plot(age_paired, mu_paired, color=COLOR_PRED, lw=0.6, alpha=0.55, zorder=4)
    axs[1].scatter(age_paired, mu_paired, s=10, color=COLOR_PRED,
                   edgecolor="white", linewidth=0.3, zorder=5)
    axs[1].set_ylabel("Predicted\nδ¹³C (‰)")

    axs[2].fill_between(age_hr, mu_hr - Z95 * sig_tot_hr, mu_hr + Z95 * sig_tot_hr,
                        color=COLOR_HR95, lw=0, zorder=1)
    axs[2].fill_between(age_hr, mu_hr - Z68 * sig_tot_hr, mu_hr + Z68 * sig_tot_hr,
                        color=COLOR_HR68, lw=0, zorder=2)
    axs[2].plot(age_hr, mu_hr, color=COLOR_HR, lw=0.55, zorder=3)
    axs[2].set_ylabel("High-resolution\nδ¹³C (‰)")
    axs[2].set_xlabel("Age (Ma)")

    for k, ax in enumerate(axs):
        ax.set_xlim(age_min - 0.03, age_max + 0.03)
        ax.set_ylim(y_min, y_max)
        ax.text(0.012, 0.94, f"({chr(97+k)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.savefig(OUT / "reconstruction_timeseries.png")
    fig.savefig(OUT / "reconstruction_timeseries.pdf")
    plt.close(fig)
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
