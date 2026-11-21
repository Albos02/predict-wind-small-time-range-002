"""Charting utilities for prediction results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from prediction_variables import CHARTS_DIR, FEATURE_STATIONS, HORIZON


def charts(
    X_test,
    Y_test,
    target,
    model,
    dontshowlist,
    Y_train=None,
    evals_result=None,
    pred_train=None,
    pred_test=None,
):
    col = f"y_{target}"
    true = Y_test[col]
    pers = X_test[target]
    pred = pd.Series(
        model.predict(X_test) + pers.to_numpy(),
        index=Y_test.index,
    )
    residual = pred - true
    abs_err = residual.abs()
    mae_p, mae_m = mean_absolute_error(true, pers), mean_absolute_error(true, pred)

    # Existing charts: model vs persistence MAE.
    if "1" not in dontshowlist:
        plt.figure(figsize=(5, 4))
        plt.bar(["persistence", "model"], [mae_p, mae_m])
        plt.title("MAE (km/h)")

    # Existing charts: predicted vs actual.
    if "2" not in dontshowlist:
        plt.figure(figsize=(6, 6))
        plt.scatter(true, pred, s=2, alpha=0.2)
        lim = [true.min(), true.max()]
        plt.plot(lim, lim, "r")
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.title(f"{target}: predicted vs actual (+{HORIZON * 10} min)")

    # Existing charts: time series, first day of test.
    if "3" not in dontshowlist:
        n = 144
        actual = true.to_numpy()[:n]
        pred_n = pred.to_numpy()[:n]
        pers_n = pers.to_numpy()[:n]
        err_n = np.abs(pred_n - actual)

        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
        ax[0].plot(actual, label="actual (t+HORIZON)")
        ax[0].plot(pred_n, label="model.predict(X_test)")
        ax[0].plot(pers_n, "--", color="gray", alpha=0.7, label="persistence (wind now)")
        ax[0].fill_between(range(n), actual, pred_n, color="red", alpha=0.25, label="error")
        ax[0].set_title(f"{target}: MAE on this extract = {err_n.mean():.2f}")
        ax[0].legend()

        ax[1].fill_between(range(n), err_n, color="red", alpha=0.5)
        ax[1].set_title("|pred - actual| (MAE = its mean)")
        fig.tight_layout()

    # Existing charts: residual histogram.
    if "4" not in dontshowlist:
        plt.figure(figsize=(6, 4))
        plt.hist(residual, bins=80)
        plt.title("Residuals (pred - actual)")

    # Existing charts: MAE by hour and month.
    if "5" not in dontshowlist:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        abs_err.groupby(abs_err.index.hour).mean().plot(ax=ax[0], title="MAE by hour")
        abs_err.groupby(abs_err.index.month).mean().plot(ax=ax[1], title="MAE by month")
        fig.tight_layout()

    # Existing charts: MAE by current speed.
    if "6" not in dontshowlist:
        bins = pd.cut(pers, [0, 5, 10, 15, 20, 30, 50, 200])
        abs_err.groupby(bins, observed=True).mean().plot.bar(
            figsize=(8, 4), title="MAE by current speed"
        )

    # Existing charts: permutation importance.
    if "7" not in dontshowlist:
        idx = np.random.RandomState(0).choice(len(X_test), min(5000, len(X_test)), replace=False)
        pi = permutation_importance(
            model,
            X_test.iloc[idx],
            (true - pers).iloc[idx],
            n_repeats=5,
            scoring="neg_mean_absolute_error",
            random_state=0,
        )
        pd.Series(pi.importances_mean, index=X_test.columns).nlargest(20)[::-1].plot.barh(
            figsize=(8, 7), title="Permutation importance (top 20)"
        )

    # Group-permutation importance ranks each feature station as a unit.
    if "10" not in dontshowlist:
        sample_idx = np.random.RandomState(0).choice(
            len(X_test), min(5000, len(X_test)), replace=False
        )
        X_sample = X_test.iloc[sample_idx]
        true_sample = true.iloc[sample_idx]
        pers_sample = pers.iloc[sample_idx].to_numpy()
        baseline_mae = mean_absolute_error(
            true_sample,
            model.predict(X_sample) + pers_sample,
        )

        rng = np.random.RandomState(42)
        station_scores = {}
        for station in FEATURE_STATIONS:
            prefix = f"{station.lower()}_"
            station_columns = [
                column for column in X_sample.columns if column.startswith(prefix)
            ]
            if not station_columns:
                continue

            shuffled_maes = []
            for _ in range(5):
                row_order = rng.permutation(len(X_sample))
                shuffled = X_sample.copy()
                shuffled.loc[:, station_columns] = X_sample.iloc[
                    row_order
                ][station_columns].to_numpy()
                shuffled_prediction = model.predict(shuffled) + pers_sample
                shuffled_maes.append(
                    mean_absolute_error(true_sample, shuffled_prediction)
                )
            station_scores[station] = np.mean(shuffled_maes) - baseline_mae

        ranked_scores = pd.Series(station_scores).sort_values()
        fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(ranked_scores))))
        colors = ["#059669" if score >= 0 else "#ef4444" for score in ranked_scores]
        bars = ax.barh(ranked_scores.index, ranked_scores.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Increase in test MAE after shuffling station features (km/h)")
        ax.set_title("Feature station usefulness ranking")
        ax.grid(axis="x", alpha=0.25)
        for bar, score in zip(bars, ranked_scores.values):
            offset = 3 if score >= 0 else -3
            ax.annotate(
                f"{score:+.3f}",
                (score, bar.get_y() + bar.get_height() / 2),
                xytext=(offset, 0),
                textcoords="offset points",
                va="center",
                ha="left" if score >= 0 else "right",
                fontsize=8,
            )
        fig.tight_layout()

    if Y_train is None or evals_result is None:
        return

    y_train = Y_train[col].to_numpy()
    y_test = true.to_numpy()
    pred_train = np.asarray(pred_train)
    pred_test = np.asarray(pred_test if pred_test is not None else pred)
    train_rmse = np.asarray(evals_result["validation_0"]["rmse"])
    test_rmse = np.asarray(evals_result["validation_1"]["rmse"])
    epochs = len(train_rmse)
    train_r2_history = 1 - train_rmse**2 / np.var(y_train)
    test_r2_history = 1 - test_rmse**2 / np.var(y_test)

    # Appended charts: training curves, residual distribution, and feature importance.
    if "8" not in dontshowlist:
        fig, ax = plt.subplots(2, 2, figsize=(18, 15))
        ax[0, 0].plot(range(epochs), train_rmse, label="Train Loss", color="blue")
        ax[0, 0].plot(range(epochs), test_rmse, label="Test Loss", color="red", linestyle="--")
        ax[0, 0].set_title("Évolution du Loss (Erreur RMSE)")
        ax[0, 0].legend()
        ax[0, 0].grid(alpha=0.3)

        ax[0, 1].plot(range(epochs), train_r2_history, label="Train R²", color="green")
        ax[0, 1].plot(range(epochs), test_r2_history, label="Test R²", color="darkorange", linestyle="--")
        r2_min = min(float(np.min(train_r2_history)), float(np.min(test_r2_history)))
        r2_max = max(float(np.max(train_r2_history)), float(np.max(test_r2_history)))
        r2_center = (r2_min + r2_max) / 2
        r2_half_range = max((r2_max - r2_min) * 0.55, 0.001)
        ax[0, 1].set_ylim(r2_center - r2_half_range, r2_center + r2_half_range)
        ax[0, 1].set_title("Focus sur la convergence (Score R²)")
        ax[0, 1].set_xlabel("Nombre d'arbres")
        ax[0, 1].set_ylabel("R² (Zoomé)")
        ax[0, 1].legend()
        ax[0, 1].grid(alpha=0.3, which="both")

        residuals = y_test - pred_test
        ax[1, 0].hist(residuals, bins=100, density=True, color="purple", alpha=0.65)
        kde_sample = residuals
        if len(kde_sample) > 5000:
            kde_sample = np.random.RandomState(42).choice(kde_sample, 5000, replace=False)
        if len(kde_sample) > 1 and np.std(kde_sample) > 0:
            kde = gaussian_kde(kde_sample)
            x_grid = np.linspace(np.min(residuals), np.max(residuals), 500)
            ax[1, 0].plot(x_grid, kde(x_grid), color="black", linewidth=1.5)
        ax[1, 0].axvline(x=0, color="black", linestyle="--")
        ax[1, 0].set_title("Distribution des erreurs (Résidus)")

        importance = model.get_booster().get_score(importance_type="weight")
        feature_names = model.get_booster().feature_names or []
        importances = [importance.get(name, 0) for name in feature_names]
        colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, max(1, len(feature_names))))
        ax[1, 1].barh(feature_names, importances, color=colors)
        ax[1, 1].set_title("Importance des variables")

        fig.tight_layout()

    if "9" not in dontshowlist:
        # Appended chart: zoomed actual-versus-predicted test series.
        n = min(300, len(y_test))
        plt.figure(figsize=(18, 6))
        plt.plot(y_test[:n], label="Réel", color="black", alpha=0.7)
        plt.plot(pred_test[:n], label="Prédit", color="orange", linestyle="--")
        plt.title("Zoom 50h : Réel vs Prédit")
        plt.legend()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    for figure_number, figure_id in enumerate(plt.get_fignums(), start=1):
        plt.figure(figure_id).savefig(
            CHARTS_DIR / f"{target}_figure_{figure_number:02d}.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()
