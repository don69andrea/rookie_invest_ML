import pandas as pd
import numpy as np
import re
from pathlib import Path
from src.common.features import require_columns
from src.schema.core_features import CORE_FEATURES

INPUT_PATH = Path("data/f2/interim/f2_results_fia_drivers_clean.csv")
OUTPUT_PATH = Path("data/f2/processed/f2_features.csv")

POINTS_TABLE = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}


def parse_time_to_seconds(s: str) -> float:
    """Konvertiert Zeitstring wie '43:01.023' oder '1:47.175' in Sekunden."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)

    s = str(s).strip()
    if s in ("", "-", "DNF", "DNS", "DSQ"):
        return np.nan

    # Reine Sekunden
    if re.match(r"^\d+(\.\d+)?$", s):
        try:
            return float(s)
        except ValueError:
            return np.nan

    parts = s.split(":")

    try:
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
            return h * 3600 + m * 60 + sec
        elif len(parts) == 2:
            m = int(parts[0])
            sec = float(parts[1])
            return m * 60 + sec
        else:
            return float(s)
    except ValueError:
        return np.nan


def parse_gap_to_seconds(s: str) -> float:
    """Konvertiert Gap in Sekunden falls möglich, ignoriert Angaben wie '1 LAP'."""
    if pd.isna(s):
        return np.nan

    s = str(s).strip()
    if s in ("", "-", "DNF", "DNS", "DSQ"):
        return np.nan

    if "LAP" in s.upper():
        return np.nan

    s_clean = s.replace("+", "")
    try:
        return float(s_clean)
    except ValueError:
        return np.nan


def mode_or_first(s: pd.Series):
    """Gibt den häufigsten Wert oder den ersten Wert zurück."""
    if s.isna().all():
        return np.nan
    m = s.mode()
    if len(m) > 0:
        return m.iloc[0]
    return s.iloc[0]


def build_f2_features() -> None:
    print(f"Lade Daten aus {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    required_cols = {
        "season",
        "race_id",
        "session",
        "laps",
        "race_time",
        "best_lap_time",
        "gap",
        "status",
        "driver_name",
        "driver_code",
        "team_name",
    }
    require_columns(df.columns, required_cols, "F2 features")

    # Harmonize season/year early so groupby can use it
    if "year" not in df.columns and "season" in df.columns:
        df = df.rename(columns={"season": "year"})

    # Serie setzen
    df["series"] = "F2"

    # Zeiten und Geschwindigkeiten vorbereiten
    print("Konvertiere Zeiten und Gaps in Sekunden ...")
    df["race_time_s"] = df["race_time"].apply(parse_time_to_seconds)
    df["best_lap_time_s"] = df["best_lap_time"].apply(parse_time_to_seconds)
    df["gap_s"] = df["gap"].apply(parse_gap_to_seconds)

    # Zielposition pro Rennen
    print("Berechne Zielposition pro Rennen ...")
    df = df.sort_values(
        ["year", "race_id", "session", "laps", "race_time_s"],
        ascending=[True, True, True, False, True],
    )
    df["finish_position"] = (
        df.groupby(["year", "race_id", "session"])
        .cumcount()
        .astype(int)
        + 1
    )

    # Einfache Punktevergabe
    print("Vergabe vereinfachter Punkte nach Zielposition ...")
    df["points"] = df["finish_position"].map(POINTS_TABLE).fillna(0.0)

    # Status Flags
    df["is_dnf_or_dq"] = df["status"].isin(["DNF", "DQ", "DSQ"])

    # Aggregation pro Fahrer und Saison
    print("Aggregiere Features pro Fahrer und Saison ...")
    group_cols = ["series", "year", "driver_code"]

    agg = (
        df.groupby(group_cols)
        .agg(
            driver_name=("driver_name", mode_or_first),
            team_name=("team_name", mode_or_first),
            n_races=("finish_position", "count"),
            total_points=("points", "sum"),
            avg_points=("points", "mean"),
            avg_finish=("finish_position", "mean"),
            best_finish=("finish_position", "min"),
            worst_finish=("finish_position", "max"),
            wins=("finish_position", lambda s: (s == 1).sum()),
            podiums=("finish_position", lambda s: (s <= 3).sum()),
            points_finishes=("finish_position", lambda s: (s <= 10).sum()),
            top10_finishes=("finish_position", lambda s: (s <= 10).sum()),
            total_laps=("laps", "sum"),
            avg_kph=("kph", "mean"),
            avg_best_lap_s=("best_lap_time_s", "mean"),
            finish_std=("finish_position", "std"),
            points_std=("points", "std"),
            dnf_count=("is_dnf_or_dq", "sum"),
        )
        .reset_index()
    )

    # Raten bauen
    agg["win_rate"] = agg["wins"] / agg["n_races"]
    agg["podium_rate"] = agg["podiums"] / agg["n_races"]
    agg["points_rate"] = agg["points_finishes"] / agg["n_races"]
    agg["top10_rate"] = agg["top10_finishes"] / agg["n_races"]
    agg["dnf_rate"] = agg["dnf_count"] / agg["n_races"]

    # Spalten in die gleiche Reihenfolge wie beim F3 Featureset bringen
    col_order = [
        "series",
        "year",
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
    agg = agg[col_order]

    # Ausgabeordner anlegen
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Speichere Features nach {OUTPUT_PATH} ...")
    agg.to_csv(OUTPUT_PATH, index=False)
    print("Fertig. Anzahl Fahrer Saison Kombinationen:", len(agg))


if __name__ == "__main__":
    build_f2_features()

    out_core = Path("data/f2/processed/f2_features.csv")
    core_df = pd.read_csv(out_core, low_memory=False)

    missing = set(CORE_FEATURES) - set(core_df.columns)
    extra = set(core_df.columns) - set(CORE_FEATURES)

    if missing:
        raise ValueError(f"F2 core schema missing columns: {sorted(missing)}")

    if extra:
        print(f"⚠️ F2 core schema extra columns (kept in file): {sorted(extra)}")

    print("✅ F2 core schema OK")
