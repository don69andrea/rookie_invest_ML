import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------------------------------
# Pfade
# -------------------------------------------------------

BASIC_PATH = Path("data/f3/processed/f3_features.csv")
ADV_PATH = Path("data/f3/processed/f3_features_advanced.csv")

PLOT_DIR = Path("plots/f3")
PLOT_DIR.mkdir(parents=True, exist_ok=True)   # Ordner automatisch anlegen


# -------------------------------------------------------
# Daten laden
# -------------------------------------------------------

def load_data():
    basic = pd.read_csv(BASIC_PATH)
    adv = pd.read_csv(ADV_PATH)

    print("Basic Season Features:", basic.shape)
    print("Advanced Season Features:", adv.shape)

    return basic, adv


# -------------------------------------------------------
# BASIC: avg_finish prüfen
# -------------------------------------------------------

def plot_basic_avg_finish(basic):
    plt.figure(figsize=(8, 5))
    basic["avg_finish"].plot(kind="hist", bins=20)
    plt.title("Verteilung: avg_finish (BASIC)")
    plt.xlabel("Durchschnittliche Zielposition")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "basic_avg_finish.png")
    plt.close()
    print("Plot gespeichert:", PLOT_DIR / "basic_avg_finish.png")


# -------------------------------------------------------
# Vergleich BASIC vs. ADVANCED: avg_finish
# -------------------------------------------------------

def plot_advanced_vs_basic_avg_finish(basic, adv):
    merged = basic.merge(
        adv[["season", "driver_name", "avg_finish"]],
        on=["season", "driver_name"],
        suffixes=("_basic", "_adv"),
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(
        merged["avg_finish_basic"],
        merged["avg_finish_adv"],
        alpha=0.4
    )
    plt.plot([0, 30], [0, 30], color="red")  # Ideallinie
    plt.title("Vergleich avg_finish BASIC vs ADVANCED")
    plt.xlabel("avg_finish_basic")
    plt.ylabel("avg_finish_advanced")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "compare_avg_finish_basic_vs_adv.png")
    plt.close()
    print("Plot gespeichert:", PLOT_DIR / "compare_avg_finish_basic_vs_adv.png")


# -------------------------------------------------------
# ADVANCED: Speed Features prüfen
# -------------------------------------------------------

def plot_advanced_speed_features(adv):
    cols = ["avg_kph", "driver_speed_mean", "team_speed_mean"]
    df = adv[cols].dropna()

    plt.figure(figsize=(10, 5))
    df.boxplot()
    plt.title("Speed Feature Verteilung (ADVANCED)")
    plt.ylabel("Wert")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "advanced_speed_boxplot.png")
    plt.close()
    print("Plot gespeichert:", PLOT_DIR / "advanced_speed_boxplot.png")


# -------------------------------------------------------
# ADVANCED: Korrelationsmatrix
# -------------------------------------------------------

def plot_adv_correlation(adv):
    cols = [
        "avg_finish",
        "avg_lap_time_s",
        "avg_kph",
        "driver_vs_team_mean",
        "lap_vs_race_avg_mean",
        "dnf_rate",
    ]

    df = adv[cols].dropna()
    corr = df.corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45)
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Map (ADVANCED)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "advanced_correlation_matrix.png")
    plt.close()
    print("Plot gespeichert:", PLOT_DIR / "advanced_correlation_matrix.png")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    basic, adv = load_data()

    plot_basic_avg_finish(basic)
    plot_advanced_vs_basic_avg_finish(basic, adv)
    plot_advanced_speed_features(adv)
    plot_adv_correlation(adv)

    print("\nAlle Plots erfolgreich generiert in:", PLOT_DIR)


if __name__ == "__main__":
    main()
