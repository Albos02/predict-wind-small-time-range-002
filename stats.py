import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from data_management import load_data, clean_data, show_new_parameters


def show_data(df):
    # print('--------------------------------------------------------------------------------------------------------------')
    # print('')
    # print('<<<<< HEAD >>>>>')
    # print('')
    # print(df.head())
    # print('')
    print('--------------------------------------------------------------------------------------------------------------')
    print('')
    print('<<<<< DF >>>>>')
    print('')
    print(df)
    print('')
    # print('--------------------------------------------------------------------------------------------------------------')
    # print('')
    # print('<<<<< SHAPE >>>>>')
    # print(df.shape)
    # print('')
    print('--------------------------------------------------------------------------------------------------------------')
    print('')
    print('<<<<< DESCRIBE >>>>>')
    print('')
    print(df.describe())
    print('')
    print('--------------------------------------------------------------------------------------------------------------')
    print('')
    print('<<<<< INFO >>>>>')
    print('')
    df.info()
    print('')
    print('--------------------------------------------------------------------------------------------------------------')
    print('')
    print('<<<<< MISSING DATA >>>>>')
    print('')
    print(df.isnull().sum())
    print('')
    print('--------------------------------------------------------------------------------------------------------------')

    print('')
    print(df[df["wind_gust_max_kmh"] > 120])
    print('')
    print('--------------------------------------------------------------------------------------------------------------')


    print('')
    print(df[df["wind_speed_kmh"] > 80])
    print('')
    print('--------------------------------------------------------------------------------------------------------------')





LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
THRESH = 15      # km/h, "windy" threshold
MIN_HOURS = 3    # hours above THRESH for a "windy day"

def show_stats(df):
    ws, gust, wd = df["wind_speed_kmh"], df["wind_gust_max_kmh"], df["wind_dir_deg"]
    avg = ws.mean()
    sector = ((wd + 22.5) % 360 // 45).astype(int)

    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(5, 3)

    # 1. per year (with legend)
    ax = fig.add_subplot(gs[0, 0])
    y_mean = ws.groupby(df.index.year).mean()
    y_gust = gust.groupby(df.index.year).max()
    b = ax.bar(y_mean.index, y_mean, label="mean speed")
    ax.set_ylabel("km/h")
    ax2 = ax.twinx()
    (l,) = ax2.plot(y_gust.index, y_gust, color="tab:red", marker="o", label="max gust")
    ax2.set_ylabel("max gust km/h")
    ax.legend([b, l], ["mean speed (left)", "max gust (right)"], loc="upper left")
    ax.set_title("Per year")

    # 2. per month + average
    ax = fig.add_subplot(gs[0, 1])
    m = ws.groupby(df.index.month).mean()
    ax.bar(m.index, m)
    ax.axhline(avg, color="red", ls="--", label=f"average {avg:.1f}")
    ax.set_xticks(range(1, 13))
    ax.set_title("Mean speed per month")
    ax.set_ylabel("km/h")
    ax.legend()

    # 3. per hour: mean vs median
    ax = fig.add_subplot(gs[0, 2])
    g = ws.groupby(df.index.hour)
    h_mean, h_med = g.mean(), g.median()
    ax.plot(h_mean.index, h_mean, marker="o", label="mean")
    ax.plot(h_med.index, h_med, marker="s", color="tab:green", label="median")
    ax.fill_between(h_mean.index, g.quantile(0.25), g.quantile(0.75), color="tab:green", alpha=0.15, label="25-75 %")
    ax.axhline(avg, color="red", ls="--", lw=0.8, label=f"overall avg {avg:.1f}")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Speed per hour of day: mean vs median")
    ax.set_ylabel("km/h")
    ax.grid(alpha=0.3)
    ax.legend()

    # 4. wind rose
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    freq = np.bincount(sector, minlength=8) / len(sector) * 100
    ax.bar(np.deg2rad(np.arange(8) * 45), freq, width=np.deg2rad(45), edgecolor="k")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(8) * 45))
    ax.set_xticklabels(LABELS)
    ax.set_title("Direction frequency (%)", pad=20)

    # 5. mean speed per direction
    ax = fig.add_subplot(gs[1, 1])
    s = ws.groupby(sector).mean().reindex(range(8))
    ax.bar(LABELS, s)
    ax.axhline(avg, color="red", ls="--")
    ax.set_title("Mean speed per direction")
    ax.set_ylabel("km/h")

    # 6. hour x month heatmap
    ax = fig.add_subplot(gs[1, 2])
    pv = ws.groupby([df.index.month, df.index.hour]).mean().unstack()
    im = ax.imshow(pv, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(12))
    ax.set_yticklabels(range(1, 13))
    ax.set_xlabel("hour")
    ax.set_ylabel("month")
    ax.set_title("Mean speed: month x hour")
    fig.colorbar(im, ax=ax, fraction=0.04, label="km/h")

    # 7. distribution
    ax = fig.add_subplot(gs[2, 0])
    bins = np.arange(0, 60, 1)
    ax.hist(ws, bins=bins, density=True, alpha=0.7, label="speed")
    ax.hist(gust, bins=bins, density=True, alpha=0.5, label="gust")
    ax.set_yscale("log")
    ax.set_title("Distribution (log y)")
    ax.set_xlabel("km/h")
    ax.legend()

    # 8. exceedance
    ax = fig.add_subplot(gs[2, 1])
    th = [5, 10, 15, 20, 30, 50]
    ax.bar([str(t) for t in th], [(ws > t).mean() * 100 for t in th])
    ax.set_title("Time with speed > threshold")
    ax.set_xlabel("km/h")
    ax.set_ylabel("%")

    # 9. gust factor
    ax = fig.add_subplot(gs[2, 2])
    gf = (gust / ws)[ws > 5]
    ax.hist(gf, bins=np.arange(1, 3, 0.02))
    ax.axvline(gf.mean(), color="red", ls="--", label=f"mean {gf.mean():.2f}")
    ax.set_title("Gust factor (gust / speed, speed > 5)")
    ax.legend()

    # 10. daily timeline + 30-day rolling mean
    ax = fig.add_subplot(gs[3, :2])
    d = ws.resample("D").mean()
    ax.plot(d.index, d, lw=0.4, alpha=0.5, label="daily mean")
    ax.plot(d.index, d.rolling(30, min_periods=10).mean(), color="red", label="30-day mean")
    ax.set_title("Mean speed over time")
    ax.set_ylabel("km/h")
    ax.legend()

    # 11. daily cycle per season
    ax = fig.add_subplot(gs[3, 2])
    seasons = {"winter": [12, 1, 2], "spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11]}
    for name, months in seasons.items():
        sub = ws[df.index.month.isin(months)]
        c = sub.groupby(sub.index.hour).mean()
        ax.plot(c.index, c, marker=".", label=name)
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Daily cycle per season")
    ax.set_ylabel("km/h")
    ax.grid(alpha=0.3)
    ax.legend()

    # 12. autocorrelation
    ax = fig.add_subplot(gs[4, 0])
    lags = list(range(1, 145, 3))
    ax.plot([l * 10 / 60 for l in lags], [ws.autocorr(l) for l in lags])
    ax.set_title("Autocorrelation (wind speed)")
    ax.set_xlabel("lag (hours)")
    ax.grid(alpha=0.3)

    # 13. correlation matrix
    ax = fig.add_subplot(gs[4, 1:])
    corr = df.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025)
    ax.set_title("Correlation")

    fig.tight_layout()
    plt.show()

    ###### OTHER SITE-SPECIFIC STATS ######

    ws, wd = df["wind_speed_kmh"], df["wind_dir_deg"]
    sector = ((wd + 22.5) % 360 // 45).astype(int)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3)

    # 6. direction x hour (% of time per hour)
    ax = fig.add_subplot(gs[0, 0])
    ct = pd.crosstab(sector, df.index.hour, normalize="columns").reindex(range(8)).fillna(0) * 100
    im = ax.imshow(ct, aspect="auto", cmap="magma", origin="lower")
    ax.set_yticks(range(8))
    ax.set_yticklabels(LABELS)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour")
    ax.set_title("Direction x hour (% of time)")
    fig.colorbar(im, ax=ax, fraction=0.04, label="%")

    # 7. speed boxplot per direction
    ax = fig.add_subplot(gs[0, 1])
    ax.boxplot([ws[sector == i] for i in range(8)], showfliers=False)
    ax.set_xticklabels(LABELS)
    ax.axhline(THRESH, color="red", ls="--", lw=0.8, label=f"{THRESH} km/h")
    ax.set_title("Speed per direction (no outliers)")
    ax.set_ylabel("km/h")
    ax.legend()

    # 8. daily temp range vs afternoon mean speed
    ax = fig.add_subplot(gs[0, 2])
    trange = df["temp_c"].resample("D").max() - df["temp_c"].resample("D").min()
    aft = ws[(df.index.hour >= 12) & (df.index.hour < 18)].resample("D").mean()
    d = pd.concat([trange, aft], axis=1, keys=["trange", "aft"]).dropna()
    hb = ax.hexbin(d["trange"], d["aft"], gridsize=40, cmap="viridis", mincnt=1, bins="log")
    k, c = np.polyfit(d["trange"], d["aft"], 1)
    x = np.array([d["trange"].min(), d["trange"].max()])
    ax.plot(x, k * x + c, color="red", label=f"fit, r={d['trange'].corr(d['aft']):.2f}")
    ax.set_xlabel("daily temp range (°C)")
    ax.set_ylabel("12-18h mean speed (km/h)")
    ax.set_title("Thermal heating vs afternoon wind")
    ax.legend()
    fig.colorbar(hb, ax=ax, fraction=0.04, label="days (log)")

    # 9. episode duration above threshold
    ax = fig.add_subplot(gs[1, 0])
    mask = ws > THRESH
    run_id = (mask != mask.shift()).cumsum()
    runs = mask.groupby(run_id).agg(["first", "size"])
    dur_h = runs.loc[runs["first"], "size"] * 10 / 60
    ax.hist(dur_h, bins=np.arange(0, 13, 0.5))
    ax.set_yscale("log")
    ax.axvline(dur_h.mean(), color="red", ls="--", label=f"mean {dur_h.mean():.1f} h")
    ax.set_xlabel("episode duration (h)")
    ax.set_title(f"Continuous runs above {THRESH} km/h")
    ax.legend()

    # 10. windy days per month (average per year)
    ax = fig.add_subplot(gs[1, 1:])
    hours = mask.resample("D").sum() / 6
    windy = (hours >= MIN_HOURS).resample("MS").sum()
    per_month = windy.groupby(windy.index.month).mean().reindex(range(1, 13))
    ax.bar(per_month.index, per_month)
    for m, v in per_month.items():
        ax.text(m, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(1, 13))
    ax.set_ylabel("days / month")
    ax.set_title(f"Windy days per month (>= {MIN_HOURS} h above {THRESH} km/h, avg over years)")

    fig.tight_layout()
    plt.show()

    ###### YEAR RANKING ######

    ws = df["wind_speed_kmh"]
    mask = ws > THRESH

    day_hours = mask.resample("D").sum() / 6
    windy = day_hours >= MIN_HOURS
    has_data = ws.resample("D").count() > 0

    y = pd.DataFrame({
        "windy_days": windy.groupby(windy.index.year).sum(),
        "days_data": has_data.groupby(has_data.index.year).sum(),
        "pct_above": mask.groupby(mask.index.year).mean() * 100,
        "mean_speed": ws.groupby(ws.index.year).mean(),
    })
    y["windy_days_365"] = y["windy_days"] / y["days_data"] * 365  # normalised for partial years
    y["partial"] = y["days_data"] < 350

    full = y[~y["partial"]]
    print(f"\nYears ranked by windy days (>= {MIN_HOURS} h above {THRESH} km/h, scaled to 365 d)\n")
    print(y.sort_values("windy_days_365", ascending=False)
            [["windy_days_365", "pct_above", "mean_speed", "days_data"]].round(1))
    print(f"\nbest : {full['windy_days_365'].idxmax()} ({full['windy_days_365'].max():.0f} days)")
    print(f"worst: {full['windy_days_365'].idxmin()} ({full['windy_days_365'].min():.0f} days)")

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    for ax, col, title, unit in [
        (axes[0], "windy_days_365", f"Windy days per year (>= {MIN_HOURS} h above {THRESH} km/h)", "days"),
        (axes[1], "pct_above", f"% of time above {THRESH} km/h", "%"),
    ]:
        avg = full[col].mean()
        colors = np.where(y[col] >= avg, "tab:green", "tab:red")
        bars = ax.bar(y.index, y[col], color=colors)
        for bar, partial in zip(bars, y["partial"]):
            if partial:
                bar.set_hatch("//")
                bar.set_alpha(0.5)
        for yr, v in y[col].items():
            ax.text(yr, v, f"{v:.0f}" if unit == "days" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
        ax.axhline(avg, color="k", ls="--", lw=0.8, label=f"average {avg:.1f}")
        ax.set_xticks(y.index)
        ax.set_xticklabels(y.index, rotation=45)
        ax.set_ylabel(unit)
        ax.set_title(title + "  (green = above avg, red = below, hatched = partial year)")
        ax.legend()

    fig.tight_layout()
    plt.show()


def main():
    df = load_data()
    df = clean_data(df, unit="kmh")
    show_data(df)
    show_new_parameters(df)
    show_stats(df)


if __name__ == "__main__":
    main()