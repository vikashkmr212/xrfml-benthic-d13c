"""
Benchmarks ten regression algorithms with 5-fold cross-validated GridSearchCV on the
paired XRF + δ¹³C data. 
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')

_HERE = Path(__file__).resolve().parent
DATA = _HERE.parent / 'data'
OUT  = _HERE.parent / 'output'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 1.2, 'axes.spines.top': False, 'axes.spines.right': False,
})

COLOR_TRAIN = '#005b96'
COLOR_TEST  = '#ffbf00'
COLOR_GOOD  = '#005b96'
COLOR_POOR  = '#bababa'

TARGET = 'C. wuellerstorfi13C'
INPUTS = ['Al_Area', 'Si_Area', 'P_Area', 'S_Area', 'Cl_Area', 'K_Area', 'Ca_Area',
          'Ti_Area', 'Cr_Area', 'Mn_Area', 'Fe_Area', 'Rh_Area', 'Cu_Area', 'Zn_Area',
          'Ga_Area', 'Br_Area', 'Rb_Area', 'Sr_Area', 'Y_Area', 'Zr_Area', 'Nb_Area',
          'Mo_Area', 'Pb_Area', 'Bi_Area', 'Ag_Area', 'Cd_Area', 'Sn_Area', 'Te_Area',
          'Ba_Area']
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


def main():
    df = pd.read_csv(DATA / 'data_clean.csv')
    X = df[INPUTS].copy()
    y = df[TARGET].copy()
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]
    # print(f"Samples after NaN drop: {len(X)}    Features: {len(INPUTS)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    # print(f"Train / Test = {len(X_train)} / {len(X_test)}")

    models_and_params = {
        'Linear Regression': {'model': LinearRegression(), 'params': {}},
        'Ridge Regression':  {'model': Ridge(),
                              'params': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}},
        'Lasso Regression':  {'model': Lasso(max_iter=5000),
                              'params': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}},
        'K-Nearest Neighbors': {'model': KNeighborsRegressor(),
                                'params': {'n_neighbors': [3, 5, 7, 9, 11, 15],
                                           'weights': ['uniform', 'distance'],
                                           'metric':  ['euclidean', 'manhattan']}},
        'Decision Tree':     {'model': DecisionTreeRegressor(random_state=RANDOM_STATE),
                              'params': {'max_depth': [3, 5, 7, 10, None],
                                         'min_samples_split': [5, 10, 20],
                                         'min_samples_leaf':  [2, 5, 10]}},
        'Random Forest':     {'model': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                              'params': {'n_estimators': [50, 100, 200],
                                         'max_depth':    [5, 10, 15, None],
                                         'min_samples_split': [5, 10],
                                         'min_samples_leaf':  [2, 5]}},
        'Extra Trees':       {'model': ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                              'params': {'n_estimators': [50, 100, 200],
                                         'max_depth':    [5, 10, 15, None],
                                         'min_samples_split': [5, 10],
                                         'min_samples_leaf':  [2, 5]}},
        'Gradient Boosting': {'model': GradientBoostingRegressor(random_state=RANDOM_STATE),
                              'params': {'n_estimators':  [50, 100, 150],
                                         'max_depth':     [3, 4, 5],
                                         'learning_rate': [0.01, 0.05, 0.1],
                                         'min_samples_split': [5, 10]}},
        'SVR (RBF)':         {'model': SVR(kernel='rbf'),
                              'params': {'C':       [0.1, 1, 10, 100],
                                         'gamma':   ['scale', 'auto', 0.01, 0.1],
                                         'epsilon': [0.01, 0.1, 0.2]}},
        'Neural Network':    {'model': MLPRegressor(random_state=RANDOM_STATE,
                                                    max_iter=1000, early_stopping=True),
                              'params': {'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                                         'alpha': [0.0001, 0.001, 0.01],
                                         'learning_rate_init': [0.001, 0.01]}},
    }

    results = []
    # print(f"\n{'Model':<25} {'CV R² (mean±std)':>18} {'Test R²':>10} {'Test RMSE':>12}")
    # print('-' * 70)

    for name, cfg in models_and_params.items():
        model, params = cfg['model'], cfg['params']
        if params:
            gs = GridSearchCV(model, params, cv=CV_FOLDS, scoring='r2',
                              n_jobs=-1, refit=True)
            gs.fit(X_train_s, y_train)
            best_model, best_params = gs.best_estimator_, gs.best_params_
        else:
            model.fit(X_train_s, y_train)
            best_model, best_params = model, {}
        cv_scores = cross_val_score(best_model, X_train_s, y_train,
                                    cv=CV_FOLDS, scoring='r2')

        yhat_train = best_model.predict(X_train_s)
        yhat_test  = best_model.predict(X_test_s)
        train_r2 = r2_score(y_train, yhat_train)
        test_r2  = r2_score(y_test, yhat_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, yhat_train))
        test_rmse  = np.sqrt(mean_squared_error(y_test, yhat_test))

        results.append({
            'Model': name, 'CV R² Mean': cv_scores.mean(), 'CV R² Std': cv_scores.std(),
            'Train R²': train_r2, 'Test R²': test_r2,
            'Train RMSE': train_rmse, 'Test RMSE': test_rmse,
            'Best Params': best_params,
            'y_train_pred': yhat_train, 'y_test_pred': yhat_test,
        })
        # print(f"{name:<25} {cv_scores.mean():>7.2f} ± {cv_scores.std():<7.2f} "
        #       f"{test_r2:>10.2f} {test_rmse:>12.2f}")

    results_df = pd.DataFrame([{k: r[k] for k in
                                ['Model', 'CV R² Mean', 'CV R² Std', 'Train R²',
                                 'Test R²', 'Train RMSE', 'Test RMSE']}
                               for r in results]
                              ).sort_values('Test R²', ascending=False).reset_index(drop=True)
    results_df.to_csv(OUT / 'model_comparison_results.csv', index=False)

    pd.DataFrame([{'Model': r['Model'], 'Test R²': r['Test R²'],
                   'Best Params': str(r['Best Params'])}
                  for r in sorted(results, key=lambda x: x['Test R²'], reverse=True)]
                 ).to_csv(OUT / 'best_hyperparameters.csv', index=False)

    sorted_results = sorted(results, key=lambda x: x['Test R²'], reverse=True)
    test_r2_values = [r['Test R²'] for r in sorted_results]
    bar_colors = [COLOR_GOOD if r > 0.75 else COLOR_POOR for r in test_r2_values]
    model_names = [r['Model'] for r in sorted_results]

    # --- Fig: bar chart of test R² ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(model_names)), test_r2_values,
                   color=bar_colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, test_r2_values):
        ax.text(val + 0.01 if val > 0 else 0.01,
                bar.get_y() + bar.get_height() / 2, f'{val:.2f}',
                va='center', ha='left', fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Test R²')
    ax.set_xlim(left=min(0, min(test_r2_values) - 0.05))
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT / 'model_comparison_barplot.png')
    fig.savefig(OUT / 'model_comparison_barplot.pdf')
    plt.close(fig)

    # --- Fig: observed-vs-predicted grid ---
    all_y = np.concatenate([y_train.values, y_test.values])
    all_pred = np.concatenate([r['y_train_pred'] for r in results] +
                              [r['y_test_pred']  for r in results])
    lo, hi = min(all_y.min(), all_pred.min()), max(all_y.max(), all_pred.max())
    margin = (hi - lo) * 0.05

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    axes = axes.flatten()
    for idx, r in enumerate(sorted_results):
        ax = axes[idx]
        ax.scatter(y_train, r['y_train_pred'], c=COLOR_TRAIN, alpha=0.7, s=25,
                   edgecolors='white', linewidth=0.1, label='Train')
        ax.scatter(y_test, r['y_test_pred'], c=COLOR_TEST, alpha=0.7, s=25,
                   edgecolors='black', linewidth=0.3, label='Test')
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], 'k--', linewidth=1)
        ax.set_xlabel('Observed δ¹³C (‰)', fontsize=9)
        ax.set_ylabel('Predicted δ¹³C (‰)', fontsize=9)
        ax.set_title(r['Model'], fontweight='bold', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.text(0.05, 0.95,
                f"Train R² = {r['Train R²']:.2f}\nTest R² = {r['Test R²']:.2f}",
                transform=ax.transAxes, fontsize=8, va='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    for idx in range(len(sorted_results), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT / 'all_models_obs_vs_pred.png')
    fig.savefig(OUT / 'all_models_obs_vs_pred.pdf')
    plt.close(fig)

    print(f"\nWrote outputs to {OUT}")


if __name__ == '__main__':
    main()
