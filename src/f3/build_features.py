from pathlib import Path
import pandas as pd
import numpy as np


INTERIM_DIR = Path("data/f3/interim")
PROCESSED_DIR = Path("data/f3/processed")


def load_f3_races_clean(path: str | Path = INTERIM_DIR / "f3_races_clean.csv") -> pd.DataFrame:
    """
    Lädt die bereinigten F3-Renndaten.
    Erwartet Spalten wie:
      - season, race_id
      - driver_name, driver_code, team_name
      - laps, kph
      - time_s, best_lap_s, gap_s
      - status
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean F3 races file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def build_f3_season_features(
    input_path: str | Path = INTERIM_DIR / "f3_races_clean.csv",
    output_path: str | Path = PROCESSED_DIR / "f3_features.csv",
) -> Path:
    """
    Erstellt Season-Level-Features für F3 auf Basis der bereits
    aufbereiteten Race-Daten (je Fahrer+Rennen eine Zeile).

    Eine Zeile im Output = ein Fahrer in einer F3-Saison.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_f3_races_clean(input_path)

    # --- Typen setzen ---

    for col in ["season", "race_id", "laps", "kph", "car_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["time_s", "best_lap_s", "gap_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strings trimmen
    for col in ["driver_name", "driver_code", "team_name", "session_type", "status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # --- Finishing Position aus Zeiten ableiten ---

    # Sortierung: pro Saison + Rennen nach time_s, danach gap_s
    df = df.sort_values(["season", "race_id", "time_s", "gap_s"])
    df["finishing_position"] = df.groupby(["season", "race_id"]).cumcount() + 1

    # --- Fahrer-Saison-Aggregation ---

    group_cols = ["season", "driver_name", "driver_code", "team_name"]
    group = df.groupby(group_cols)

    agg = group.agg(
        n_races=("race_id", "nunique"),
        avg_finish=("finishing_position", "mean"),
        best_finish=("finishing_position", "min"),
        worst_finish=("finishing_position", "max"),
        wins=("finishing_position", lambda s: (s == 1).sum()),
        podiums=("finishing_position", lambda s: (s <= 3).sum()),
        top10_finishes=("finishing_position", lambda s: (s <= 10).sum()),
        total_laps=("laps", "sum"),
        avg_kph=("kph", "mean"),
        avg_best_lap_s=("best_lap_s", "mean"),
        finish_std=("finishing_position", "std"),
    ).reset_index()

    # --- Raten berechnen ---

    agg["win_rate"] = agg["wins"] / agg["n_races"]
    agg["podium_rate"] = agg["podiums"] / agg["n_races"]
    agg["top10_rate"] = agg["top10_finishes"] / agg["n_races"]

    # Punkte & DNF analog F2 erstmal als Platzhalter
    agg["total_points"] = np.nan
    agg["avg_points"] = np.nan
    agg["points_finishes"] = np.nan
    agg["points_rate"] = np.nan
    agg["dnf_count"] = np.nan
    agg["dnf_rate"] = np.nan
    agg["points_std"] = np.nan

    # Serienlabel
    agg["series"] = "F3"

    # --- Spaltenreihenfolge (so gut wie möglich kompatibel zu F1/F2) ---

    col_order = [
        "series",
        "season",          # entspricht year
        "driver_name",
        "driver_code",
        "team_name",
        "n_races",
        "total_points",
        "avg_points",
        "avg_finish",
        "best_finish",
        "worst_finish",
        "wins",
        "win_rate",
        "podiums",
        "podium_rate",
        "points_finishes",
        "points_rate",
        "top10_finishes",
        "top10_rate",
        "total_laps",
        "avg_kph",
        "avg_best_lap_s",
        "finish_std",
        "points_std",
        "dnf_count",
        "dnf_rate",
    ]

    col_order = [c for c in col_order if c in agg.columns]
    agg = agg[col_order].copy()

    # Sortierung für Lesbarkeit
    agg = agg.sort_values(["season", "driver_name"]).reset_index(drop=True)

    # Harmonize column name for cross-series merge
    if "season" in agg.columns and "year" not in agg.columns:
        agg = agg.rename(columns={"season": "year"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)
    print(f"✅ F3 season features written to: {output_path}")

    return output_path


if __name__ == "__main__":
    build_f3_season_features()
