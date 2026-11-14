"""
Comprehensive Evaluation and Diagnostics Suite for 20-minute Wind Prediction.

Generates in-depth visual diagnostics and analysis charts:
  1. Model Comparison & Residual Distributions
  2. Performance by Wind Speed Regime & Calibration Curves
  3. Diurnal (Hourly) & Seasonal (Monthly) Patterns
  4. Directional Sector & Wind Rose Analysis
  5. Dynamic Episode Case Studies (Ramp-ups, Storms, Thermal Breezes)
  6. Feature Importance & Model Interpretation
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb

from predict_wind import (
    load_and_preprocess_data,
    create_features_and_target,
    predict_with_artifact,
    MODEL_PATH,
)
from sklearn.impute import SimpleImputer


CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

# Styling configuration for publication-quality charts
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.dpi": 150,
})


def run_full_evaluation():
    print("=" * 70)
    print("  RUNNING 20-MIN WIND PREDICTION DIAGNOSTIC SUITE")
    print("=" * 70)

    df = load_and_preprocess_data()
    X, y = create_features_and_target(df)

    test_year = 2025
    train_mask = X.index < "2024-01-01"
    val_mask = (X.index >= "2024-01-01") & (X.index < f"{test_year}-01-01")
    test_mask = X.index >= f"{test_year}-01-01"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nTrain samples: {len(X_train):,} (2004-2023)")
    print(f"Val samples  : {len(X_val):,} (2024)")
    print(f"Test samples : {len(X_test):,} (2025)")

    # 1. Models & Predictions
    print("\nTraining / Loading models...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Ridge
    ridge = Ridge(alpha=1.0).fit(X_train_imp, y_train)
    preds_ridge = ridge.predict(X_test_imp)

    # LightGBM
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        lgbm = artifact["model"]
    else:
        from predict_wind import train_quantile_model
        delta_train = y_train - X_train["speed_t0"]
        delta_val = y_val - X_val["speed_t0"]
        quantile_models = {}
        for q in [0.1, 0.5, 0.9]:
            quantile_models[q] = train_quantile_model(X_train, delta_train, X_val, delta_val, alpha=q)
        artifact = {
            "model": quantile_models[0.5],
            "quantile_models": quantile_models,
            "feature_names": X_train.columns.tolist(),
            "is_delta_model": True,
            "imputer": imputer,
            "ridge_model": ridge,
        }
        joblib.dump(artifact, MODEL_PATH)
        lgbm = artifact["model"]

    preds_lgb_dict = predict_with_artifact(artifact, X_test)
    preds_lgb = preds_lgb_dict["p50"]
    preds_lgb_p10 = preds_lgb_dict.get("p10", preds_lgb)
    preds_lgb_p90 = preds_lgb_dict.get("p90", preds_lgb)

    # Persistence
    preds_persist = X_test["speed_t0"].values

    # Assemble test evaluation dataframe
    eval_df = pd.DataFrame({
        "y_true": y_test,
        "pred_lgb": preds_lgb,
        "pred_lgb_p10": preds_lgb_p10,
        "pred_lgb_p90": preds_lgb_p90,
        "pred_ridge": preds_ridge,
        "pred_persist": preds_persist,
        "speed_t0": X_test["speed_t0"],
        "speed_diff_10m": X_test["speed_diff_10m"],
        "gust_t0": X_test["gust_t0"],
        "wind_dir": df.loc[X_test.index, "dkl010z0"],
        "hour": X_test.index.hour,
        "month": X_test.index.month,
    }, index=X_test.index)

    # --- CHART 1: Model Comparison, Residuals, and Cumulative Error ---
    plot_model_comparison(eval_df)

    # --- CHART 2: Speed Regimes, Calibration & Bias ---
    plot_speed_regimes_and_calibration(eval_df)

    # --- CHART 3: Diurnal & Seasonal Patterns ---
    plot_diurnal_and_seasonal(eval_df)

    # --- CHART 4: Directional Sectors & Wind Rose ---
    plot_directional_analysis(eval_df)

    # --- CHART 5: Time-Series Case Studies ---
    plot_event_case_studies(eval_df)

    # --- CHART 6: Feature Importances ---
    plot_feature_importance(lgbm, ridge, X_train.columns.tolist())

    print("\n" + "=" * 70)
    print(f"  All 6 diagnostic charts successfully generated in '{CHARTS_DIR}/'")
    print("=" * 70)


def plot_model_comparison(eval_df: pd.DataFrame):
    print("Generating Chart 1: Model Comparison & Residual Distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Overall Metrics
    models = ["Persistence", "Ridge Regression", "LightGBM"]
    preds = [eval_df["pred_persist"], eval_df["pred_ridge"], eval_df["pred_lgb"]]
    maes = [mean_absolute_error(eval_df["y_true"], p) for p in preds]
    rmses = [root_mean_squared_error(eval_df["y_true"], p) for p in preds]

    x = np.arange(len(models))
    width = 0.35
    axes[0].bar(x - width/2, maes, width, label="MAE", color="#3b82f6", alpha=0.9)
    axes[0].bar(x + width/2, rmses, width, label="RMSE", color="#ef4444", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Error (m/s)")
    axes[0].set_title("A. Error Metrics Comparison (2025 Test)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    for i in range(len(models)):
        axes[0].text(i - width/2, maes[i] + 0.02, f"{maes[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        axes[0].text(i + width/2, rmses[i] + 0.02, f"{rmses[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, max(rmses) * 1.18)

    # Panel B: Residual Distributions (y_true - y_pred)
    res_persist = eval_df["y_true"] - eval_df["pred_persist"]
    res_lgb = eval_df["y_true"] - eval_df["pred_lgb"]
    bins = np.linspace(-4, 4, 100)
    axes[1].hist(res_persist, bins=bins, density=True, alpha=0.45, color="#64748b", label=f"Persistence (std={res_persist.std():.2f})")
    axes[1].hist(res_lgb, bins=bins, density=True, alpha=0.6, color="#059669", label=f"LightGBM (std={res_lgb.std():.2f})")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual Error ($y_{true} - \\hat{y}$) [m/s]")
    axes[1].set_ylabel("Density")
    axes[1].set_title("B. Residual Error Distribution (Tighter is Better)")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(-3.5, 3.5)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel C: Cumulative Absolute Error Curves
    err_persist = np.sort(np.abs(res_persist))
    err_lgb = np.sort(np.abs(res_lgb))
    err_ridge = np.sort(np.abs(eval_df["y_true"] - eval_df["pred_ridge"]))
    y_pct = np.linspace(0, 100, len(err_persist))

    axes[2].plot(err_persist, y_pct, label="Persistence", color="#64748b", linestyle="--", linewidth=2)
    axes[2].plot(err_ridge, y_pct, label="Ridge Regression", color="#3b82f6", linewidth=2)
    axes[2].plot(err_lgb, y_pct, label="LightGBM", color="#059669", linewidth=2.5)
    axes[2].axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    axes[2].axvline(1.0, color="gray", linestyle=":", alpha=0.7)
    axes[2].text(0.52, 20, "0.5 m/s", color="gray", fontsize=9)
    axes[2].text(1.02, 20, "1.0 m/s", color="gray", fontsize=9)
    axes[2].set_xlim(0, 2.5)
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel("Absolute Error Threshold (m/s)")
    axes[2].set_ylabel("% Predictions within Threshold")
    axes[2].set_title("C. Cumulative Error Distribution")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_model_comparison_and_residuals.png")
    plt.close()


def plot_speed_regimes_and_calibration(eval_df: pd.DataFrame):
    print("Generating Chart 2: Speed Regimes & Calibration Curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: MAE by Speed Regime
    bins = [0, 2, 4, 6, 8, 30]
    labels = ["0-2\n(Calm)", "2-4\n(Light)", "4-6\n(Mod.)", "6-8\n(Fresh)", "8+\n(Strong)"]
    eval_df["regime"] = pd.cut(eval_df["y_true"], bins=bins, labels=labels, right=False)

    reg_stats = []
    for label in labels:
        sub = eval_df[eval_df["regime"] == label]
        if len(sub) == 0:
            continue
        mae_p = mean_absolute_error(sub["y_true"], sub["pred_persist"])
        mae_l = mean_absolute_error(sub["y_true"], sub["pred_lgb"])
        reg_stats.append({
            "regime": label,
            "count": len(sub),
            "mae_p": mae_p,
            "mae_l": mae_l,
            "pct": len(sub) / len(eval_df) * 100,
            "diff_pct": (mae_p - mae_l) / mae_p * 100,
        })
    rdf = pd.DataFrame(reg_stats)

    x = np.arange(len(rdf))
    width = 0.35
    axes[0].bar(x - width/2, rdf["mae_p"], width, label="Persistence", color="#94a3b8")
    axes[0].bar(x + width/2, rdf["mae_l"], width, label="LightGBM", color="#059669")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r}\n({p:.1f}%)" for r, p in zip(rdf["regime"], rdf["pct"])])
    axes[0].set_ylabel("MAE (m/s)")
    axes[0].set_title("A. MAE by Wind Speed Regime")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    for i in range(len(rdf)):
        diff = rdf.loc[i, 'diff_pct']
        color = "green" if diff > 0 else "red"
        axes[0].text(i, max(rdf.loc[i, 'mae_p'], rdf.loc[i, 'mae_l']) + 0.08,
                     f"{diff:+.1f}%", ha="center", fontsize=9, fontweight="bold", color=color)
    axes[0].set_ylim(0, rdf["mae_l"].max() * 1.2)

    # Panel B: Calibration / Actual vs Predicted Hexbin
    hb = axes[1].hexbin(eval_df["y_true"], eval_df["pred_lgb"], gridsize=40, cmap="Blues", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=axes[1])
    cb.set_label("Log Count")

    # Conditional mean curve
    y_bins = np.linspace(0, 15, 16)
    binned = eval_df.groupby(pd.cut(eval_df["y_true"], y_bins, right=False), observed=False)
    mean_true = binned["y_true"].mean()
    mean_pred = binned["pred_lgb"].mean()
    axes[1].plot([0, 16], [0, 16], "r--", linewidth=2, label="Ideal (y = x)")
    axes[1].plot(mean_true, mean_pred, "k-o", linewidth=2, markersize=5, label="Model Conditional Mean")
    axes[1].set_xlim(0, 14)
    axes[1].set_ylim(0, 14)
    axes[1].set_xlabel("Actual Wind Speed at $t+20\\text{m}$ (m/s)")
    axes[1].set_ylabel("Predicted Wind Speed (m/s)")
    axes[1].set_title("B. Prediction Calibration & Spread")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel C: Error by Dynamic Acceleration Rate
    eval_df["diff_cat"] = pd.cut(
        eval_df["speed_diff_10m"],
        bins=[-np.inf, -1.0, -0.2, 0.2, 1.0, np.inf],
        labels=["Fast Decel\n(< -1.0)", "Mild Decel\n(-1.0 to -0.2)", "Steady\n(-0.2 to +0.2)", "Mild Accel\n(+0.2 to +1.0)", "Fast Accel\n(> +1.0)"]
    )
    diff_stats = []
    for cat in eval_df["diff_cat"].cat.categories:
        sub = eval_df[eval_df["diff_cat"] == cat]
        mae_p = mean_absolute_error(sub["y_true"], sub["pred_persist"])
        mae_l = mean_absolute_error(sub["y_true"], sub["pred_lgb"])
        diff_stats.append({"cat": cat, "mae_p": mae_p, "mae_l": mae_l, "diff": (mae_p - mae_l) / mae_p * 100})
    ddf = pd.DataFrame(diff_stats)

    x2 = np.arange(len(ddf))
    axes[2].bar(x2 - width/2, ddf["mae_p"], width, label="Persistence", color="#94a3b8")
    axes[2].bar(x2 + width/2, ddf["mae_l"], width, label="LightGBM", color="#059669")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(ddf["cat"])
    axes[2].set_ylabel("MAE (m/s)")
    axes[2].set_title("C. Performance during Wind Acceleration")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    for i in range(len(ddf)):
        diff = ddf.loc[i, 'diff']
        axes[2].text(i, max(ddf.loc[i, 'mae_p'], ddf.loc[i, 'mae_l']) + 0.04,
                     f"+{diff:.1f}%", ha="center", fontsize=9, fontweight="bold", color="green")
    axes[2].set_ylim(0, ddf["mae_p"].max() * 1.18)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_speed_regimes_and_calibration.png")
    plt.close()


def plot_diurnal_and_seasonal(eval_df: pd.DataFrame):
    print("Generating Chart 3: Diurnal & Seasonal Patterns...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Diurnal (Hourly)
    hourly = eval_df.groupby("hour").apply(lambda g: pd.Series({
        "mae_p": mean_absolute_error(g["y_true"], g["pred_persist"]),
        "mae_l": mean_absolute_error(g["y_true"], g["pred_lgb"]),
        "avg_speed": g["y_true"].mean()
    }), include_groups=False)

    axes[0].plot(hourly.index, hourly["mae_p"], "o--", color="#94a3b8", label="Persistence MAE", linewidth=2)
    axes[0].plot(hourly.index, hourly["mae_l"], "s-", color="#059669", label="LightGBM MAE", linewidth=2.5)
    axes[0].set_xlabel("Hour of Day (UTC)")
    axes[0].set_ylabel("MAE (m/s)")
    axes[0].set_title("A. Hourly Diurnal Error Profile")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Twin axis for average wind speed
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(hourly.index, hourly["avg_speed"], ":", color="#f59e0b", label="Avg Speed", linewidth=1.8)
    ax0_twin.set_ylabel("Average Wind Speed (m/s)", color="#b45309")
    ax0_twin.tick_params(axis="y", labelcolor="#b45309")
    ax0_twin.grid(False)

    # Panel B: Seasonal (Monthly)
    monthly = eval_df.groupby("month").apply(lambda g: pd.Series({
        "mae_p": mean_absolute_error(g["y_true"], g["pred_persist"]),
        "mae_l": mean_absolute_error(g["y_true"], g["pred_lgb"]),
        "avg_speed": g["y_true"].mean()
    }), include_groups=False)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[1].plot(monthly.index, monthly["mae_p"], "o--", color="#94a3b8", label="Persistence", linewidth=2)
    axes[1].plot(monthly.index, monthly["mae_l"], "s-", color="#059669", label="LightGBM", linewidth=2.5)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(months)
    axes[1].set_xlabel("Month of Year")
    axes[1].set_ylabel("MAE (m/s)")
    axes[1].set_title("B. Monthly Seasonal Error Profile")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel C: 2D Heatmap of MAE (Month x Hour)
    grid_mae = np.zeros((12, 24))
    for m in range(1, 13):
        for h in range(24):
            sub = eval_df[(eval_df["month"] == m) & (eval_df["hour"] == h)]
            grid_mae[m-1, h] = mean_absolute_error(sub["y_true"], sub["pred_lgb"])

    im = axes[2].imshow(grid_mae, cmap="YlOrRd", aspect="auto", origin="lower")
    cb = fig.colorbar(im, ax=axes[2])
    cb.set_label("LightGBM MAE (m/s)")
    axes[2].set_yticks(range(12))
    axes[2].set_yticklabels(months)
    axes[2].set_xticks(range(0, 24, 4))
    axes[2].set_xlabel("Hour of Day (UTC)")
    axes[2].set_ylabel("Month")
    axes[2].set_title("C. Error Heatmap (Month vs Hour)")

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_diurnal_and_seasonal_patterns.png")
    plt.close()


def plot_directional_analysis(eval_df: pd.DataFrame):
    print("Generating Chart 4: Wind Direction & Sector Analysis...")
    fig = plt.figure(figsize=(18, 5.5))

    # Panel A: Polar Radar Plot of MAE by 16 Sectors
    ax1 = fig.add_subplot(131, projection="polar")
    n_sectors = 16
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
    deg_step = 360 / n_sectors
    sector_bins = np.linspace(0, 360, n_sectors + 1)
    eval_df["dir_16"] = pd.cut(eval_df["wind_dir"] % 360, bins=sector_bins, include_lowest=True)

    mae_p_sectors = []
    mae_l_sectors = []
    for cat in eval_df["dir_16"].cat.categories:
        sub = eval_df[eval_df["dir_16"] == cat]
        if len(sub) > 0:
            mae_p_sectors.append(mean_absolute_error(sub["y_true"], sub["pred_persist"]))
            mae_l_sectors.append(mean_absolute_error(sub["y_true"], sub["pred_lgb"]))
        else:
            mae_p_sectors.append(0)
            mae_l_sectors.append(0)

    # Close the polar loop
    angles_closed = np.concatenate((angles, [angles[0]]))
    mae_p_closed = np.concatenate((mae_p_sectors, [mae_p_sectors[0]]))
    mae_l_closed = np.concatenate((mae_l_sectors, [mae_l_sectors[0]]))

    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)  # Clockwise like a compass
    ax1.plot(angles_closed, mae_p_closed, color="#94a3b8", linestyle="--", linewidth=1.8, label="Persistence")
    ax1.plot(angles_closed, mae_l_closed, color="#059669", linewidth=2.2, label="LightGBM")
    ax1.fill(angles_closed, mae_l_closed, color="#059669", alpha=0.15)
    ax1.set_title("A. MAE by Compass Direction", pad=15)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.25, -0.1))

    # Panel B: Major Named Wind Regimes on Lake Geneva
    ax2 = fig.add_subplot(132)
    regime_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    regime_labels = ["N", "NE\n(Bise)", "E", "SE\n(Vaudaire)", "S", "SW\n(Vent)", "W", "NW"]
    eval_df["regime_8"] = pd.cut(eval_df["wind_dir"] % 360, bins=regime_bins, labels=regime_labels, include_lowest=True)

    stats8 = []
    for lbl in regime_labels:
        sub = eval_df[eval_df["regime_8"] == lbl]
        mae_p = mean_absolute_error(sub["y_true"], sub["pred_persist"])
        mae_l = mean_absolute_error(sub["y_true"], sub["pred_lgb"])
        stats8.append({"sector": lbl, "mae_p": mae_p, "mae_l": mae_l, "diff": (mae_p - mae_l)/mae_p * 100})
    df8 = pd.DataFrame(stats8)

    x = np.arange(len(df8))
    width = 0.35
    ax2.bar(x - width/2, df8["mae_p"], width, label="Persistence", color="#94a3b8")
    ax2.bar(x + width/2, df8["mae_l"], width, label="LightGBM", color="#059669")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df8["sector"])
    ax2.set_ylabel("MAE (m/s)")
    ax2.set_title("B. MAE by Major Lake Geneva Wind Regime")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Wind Frequency by Direction
    ax3 = fig.add_subplot(133, projection="polar")
    counts, _ = np.histogram(eval_df["wind_dir"].dropna() % 360, bins=sector_bins)
    freqs = counts / len(eval_df) * 100
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction(-1)
    bars = ax3.bar(angles, freqs, width=np.deg2rad(deg_step), color="#3b82f6", alpha=0.7, edgecolor="white")
    ax3.set_title("C. BOU Station Wind Rose (% Occurrences)", pad=15)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_directional_analysis.png")
    plt.close()


def plot_event_case_studies(eval_df: pd.DataFrame):
    print("Generating Chart 5: Dynamic Event Case Studies...")
    fig, axes = plt.subplots(3, 1, figsize=(18, 10))

    episodes = [
        ("Event 1: Severe Autumn Gale & Sudden Peak (Oct 23, 2025)", "2025-10-23 00:00", "2025-10-24 06:00"),
        ("Event 2: Sustained Storm Front (Oct 04, 2025)", "2025-10-04 06:00", "2025-10-05 06:00"),
        ("Event 3: Summer Thermal Breeze Cycle (July 15-16, 2025)", "2025-07-15 00:00", "2025-07-16 23:50"),
    ]

    for ax, (title, start_ts, end_ts) in zip(axes, episodes):
        sub = eval_df.loc[start_ts:end_ts]
        if len(sub) == 0:
            continue
        ax.fill_between(
            sub.index,
            sub["pred_lgb_p10"],
            sub["pred_lgb_p90"],
            color="#10b981",
            alpha=0.22,
            label="80% Prediction Interval [p10, p90] (Pinball Loss)",
        )
        ax.plot(sub.index, sub["y_true"], label="Actual Speed at t+20m", color="#0f172a", linewidth=2.2)
        ax.plot(sub.index, sub["pred_lgb"], label="LightGBM Median Forecast (p50)", color="#059669", linewidth=2, linestyle="-")
        ax.plot(sub.index, sub["pred_persist"], label="Persistence Baseline (t)", color="#94a3b8", linewidth=1.5, linestyle="--")

        mae_l = mean_absolute_error(sub["y_true"], sub["pred_lgb"])
        mae_p = mean_absolute_error(sub["y_true"], sub["pred_persist"])

        ax.set_title(f"{title} | LightGBM MAE: {mae_l:.2f} m/s vs Persistence: {mae_p:.2f} m/s ({((mae_p-mae_l)/mae_p*100):+.1f}%)")
        ax.set_ylabel("Wind Speed (m/s)")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Timestamp")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_event_case_studies.png")
    plt.close()


def plot_feature_importance(lgbm_model, ridge_model, feature_names):
    print("Generating Chart 6: Feature Importances & Model Drivers...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    # Panel A: LightGBM Importance (Gain & Splits)
    importance_split = lgbm_model.feature_importances_
    importance_gain = lgbm_model.booster_.feature_importance(importance_type="gain")

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "split": importance_split,
        "gain": importance_gain
    }).sort_values("split", ascending=True)

    y_pos = np.arange(len(imp_df))
    axes[0].barh(y_pos, imp_df["split"], color="#059669", alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(imp_df["feature"])
    axes[0].set_xlabel("Number of Decision Tree Splits")
    axes[0].set_title("A. LightGBM Feature Importance (Split Frequency)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel B: Ridge Regression Coefficients
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": ridge_model.coef_
    }).sort_values("coef", ascending=True)

    y_pos2 = np.arange(len(coef_df))
    colors = ["#ef4444" if c < 0 else "#3b82f6" for c in coef_df["coef"]]
    axes[1].barh(y_pos2, coef_df["coef"], color=colors, alpha=0.85)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_yticks(y_pos2)
    axes[1].set_yticklabels(coef_df["feature"])
    axes[1].set_xlabel("Standardized Linear Model Coefficient Value")
    axes[1].set_title("B. Ridge Regression Coefficients (Directional Driver)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_feature_importance.png")
    plt.close()


if __name__ == "__main__":
    run_full_evaluation()
