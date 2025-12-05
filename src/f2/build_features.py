from pathlib import Path
import pandas as pd
import numpy as np


INTERIM_DIR = Path("data/f2/interim")
PROCESSED_DIR = Path("data/f2/processed")


def load_f2_results_clean(path: str | Path = INTERIM_DIR / "f2_race_results_clean.csv") -> pd.DataFrame:
    """
    Lädt die bereinigten F2-Rennergebnisse.
    Erwartet Spalten wie:
      - race_date
      - circuit_name, race_type
      - position, laps, kph
      - driver_name, team_name
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean F2 race results file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def build_f2_season_features(
    input_path: str | Path = INTERIM_DIR / "f2_race_results_clean.csv",
    output_path: str | Path = PROCESSED_DIR / "f2_features.csv",
) -> Path:
    """
    Erstellt Season-Level-Features für F2 auf Basis der bereinigten
    Rennresultate (Feature + Sprint Races).

    Eine Zeile im Output = ein Fahrer in einer Saison.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_f2_results_clean(input_path)

    # --- Grundaufbereitung ---

    # Datum → Jahr
    if "race_date" not in df.columns:
        raise ValueError("Spalte 'race_date' wird für F2-Features erwartet.")

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["year"] = df["race_date"].dt.year.astype("Int64")

    # Strings trimmen
    for col in ["driver_name", "team_name", "circuit_name", "race_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Numerische Spalten
    for col in ["position", "laps", "car_number", "kph", "track_length_km"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Punkte: in deinen F2-Rohdaten gibt es (noch) keine Punktespalte.
    # Wir legen hier 'points' als NaN an – damit die Spalten-Struktur
    # kompatibel zu F1 bleibt, aber Werte später gezielt ergänzt werden können.
    if "points" not in df.columns:
        df["points"] = np.nan

    # --- Fahrer-Saison-Gruppierung ---

    group_cols = ["year", "driver_name", "team_name"]
    group = df.groupby(group_cols)

    agg = group.agg(
        n_races=("circuit_name", "count"),            # Anzahl Rennen (Feature + Sprint)
        total_points=("points", "sum"),
        avg_points=("points", "mean"),
        avg_finish=("position", "mean"),
        best_finish=("position", "min"),
        worst_finish=("position", "max"),
        wins=("position", lambda s: (s == 1).sum()),
        podiums=("position", lambda s: (s <= 3).sum()),
        points_finishes=("points", lambda s: (s > 0).sum()),
        top10_finishes=("position", lambda s: (s <= 10).sum()),
        total_laps=("laps", "sum"),
        avg_kph=("kph", "mean"),
        finish_std=("position", "std"),
        points_std=("points", "std"),
    ).reset_index()

    # --- Raten berechnen ---

    agg["win_rate"] = agg["wins"] / agg["n_races"]
    agg["podium_rate"] = agg["podiums"] / agg["n_races"]
    agg["points_rate"] = agg["points_finishes"] / agg["n_races"]
    agg["top10_rate"] = agg["top10_finishes"] / agg["n_races"]

    # F2 hat keine saubere DNF-Codierung in diesen Daten → als Platzhalter NaN
    agg["dnf_count"] = np.nan
    agg["dnf_rate"] = np.nan

    # Serienlabel
    agg["series"] = "F2"

    # --- Spaltenreihenfolge (ähnlich F1, soweit möglich) ---

    col_order = [
        "series",
        "year",
        "driver_name",
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
        "finish_std",
        "points_std",
        "dnf_count",
        "dnf_rate",
    ]

    # Nur Spalten nehmen, die es wirklich gibt
    col_order = [c for c in col_order if c in agg.columns]
    agg = agg[col_order].copy()

    # Sortierung für Lesbarkeit
    agg = agg.sort_values(["year", "driver_name"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)
    print(f"✅ F2 season features written to: {output_path}")

    return output_path


if __name__ == "__main__":
    build_f2_season_features()
