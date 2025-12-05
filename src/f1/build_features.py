from pathlib import Path
import pandas as pd


INTERIM_DIR = Path("data/f1/interim")
PROCESSED_DIR = Path("data/f1/processed")


def load_f1_clean(path: str | Path = INTERIM_DIR / "f1_race_driver_clean.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean F1 race-driver file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def build_f1_season_features(
    input_path: str | Path = INTERIM_DIR / "f1_race_driver_clean.csv",
    output_path: str | Path = PROCESSED_DIR / "f1_features.csv",
) -> Path:
    """
    Erstellt Season-Level-Features für F1:
    - Aggregation von Fahrer-Rennen zu Fahrer-Saison
    - Berechnung von Performance-, Pace-, Konsistenz-, DNF- und Team-Features
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_f1_clean(input_path)

    # --- Grundtypen setzen ---

    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
    df["round"] = pd.to_numeric(df.get("round"), errors="coerce").astype("Int64")

    for col in ["grid_position", "finishing_position", "finishing_order", "laps_completed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["points"] = pd.to_numeric(df.get("points"), errors="coerce")
    df["result_ms"] = pd.to_numeric(df.get("result_ms"), errors="coerce")
    df["fastest_lap_speed"] = pd.to_numeric(df.get("fastest_lap_speed"), errors="coerce")

    # Positionsveränderung: Start vs. Endposition (Order, damit auch DNFs enthalten sind)
    df["pos_change"] = df["grid_position"] - df["finishing_order"]

    # Safety: Flags sicherstellen
    if "status_text" in df.columns:
        df["is_classified_finish"] = df["status_text"].str.contains("Finished", na=False)
        df["is_dnf"] = ~df["is_classified_finish"]
    else:
        df["is_dnf"] = False

    df["is_points_finish"] = df["points"].fillna(0) > 0

    # Nur Zeilen mit gültigem Jahr / Fahrer
    df_season = df.dropna(subset=["year", "driver_id"]).copy()

    # --- Team-Aggregation pro Jahr & Team ---

    team_group = df_season.groupby(["year", "constructor_id"])
    team_agg = team_group.agg(
        team_total_points=("points", "sum"),
        team_avg_points=("points", "mean"),
        team_avg_pos_season=("finishing_order", "mean"),
        team_speed=("fastest_lap_speed", "mean"),
        team_n_races=("race_id", "nunique"),
    ).reset_index()

    # --- Fahrer-Saison-Aggregation ---

    driver_group = df_season.groupby(["year", "driver_id"])
    driver_agg = driver_group.agg(
        n_races=("race_id", "nunique"),
        total_points=("points", "sum"),
        avg_points=("points", "mean"),
        avg_grid=("grid_position", "mean"),
        avg_finish=("finishing_order", "mean"),
        best_finish=("finishing_order", "min"),
        worst_finish=("finishing_order", "max"),
        wins=("finishing_order", lambda s: (s == 1).sum()),
        podiums=("finishing_order", lambda s: (s <= 3).sum()),
        points_finishes=("is_points_finish", lambda s: s.fillna(False).sum()),
        top10_finishes=("finishing_order", lambda s: (s <= 10).sum()),
        total_laps=("laps_completed", "sum"),
        avg_kph=("fastest_lap_speed", "mean"),
        finish_std=("finishing_order", "std"),
        points_std=("points", "std"),
        avg_pos_change=("pos_change", "mean"),
        pos_change_std=("pos_change", "std"),
        dnf_count=("is_dnf", lambda s: s.fillna(False).sum()),
    ).reset_index()

    # Raten aus den Counts ableiten
    driver_agg["win_rate"] = driver_agg["wins"] / driver_agg["n_races"]
    driver_agg["podium_rate"] = driver_agg["podiums"] / driver_agg["n_races"]
    driver_agg["points_rate"] = driver_agg["points_finishes"] / driver_agg["n_races"]
    driver_agg["top10_rate"] = driver_agg["top10_finishes"] / driver_agg["n_races"]
    driver_agg["dnf_rate"] = driver_agg["dnf_count"] / driver_agg["n_races"]

    # --- Meta-Infos je Fahrer/Saison: Name, Code, Nationalität, Team ---

    meta_cols = [
        "driver_name",
        "driver_code",
        "driver_nationality",
        "constructor_id",
        "constructor_name",
    ]

    meta = (
        df_season
        .sort_values(["year", "driver_id", "race_id"])
        .groupby(["year", "driver_id"])[meta_cols]
        .first()
        .reset_index()
    )

    season = driver_agg.merge(meta, on=["year", "driver_id"], how="left")
    season = season.merge(team_agg, on=["year", "constructor_id"], how="left")

    # --- Fahrer vs Team Deltas ---

    season["driver_speed"] = season["avg_kph"]
    season["driver_vs_team_speed"] = season["driver_speed"] - season["team_speed"]
    season["driver_vs_team_avg_finish"] = season["team_avg_pos_season"] - season["avg_finish"]
    season["driver_vs_team_avg_points"] = season["avg_points"] - season["team_avg_points"]

    # Serie kennzeichnen
    season["series"] = "F1"

    # Ordnung der Spalten (optional, aber nice)
    col_order = [
        "series",
        "year",
        "driver_id",
        "driver_name",
        "driver_code",
        "driver_nationality",
        "constructor_id",
        "constructor_name",
        "n_races",
        "total_points",
        "avg_points",
        "avg_grid",
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
        "avg_pos_change",
        "pos_change_std",
        "dnf_count",
        "dnf_rate",
        "team_total_points",
        "team_avg_points",
        "team_avg_pos_season",
        "team_speed",
        "team_n_races",
        "driver_speed",
        "driver_vs_team_speed",
        "driver_vs_team_avg_finish",
        "driver_vs_team_avg_points",
    ]

    # Nur Spalten verwenden, die es wirklich gibt
    col_order = [c for c in col_order if c in season.columns]
    season = season[col_order].copy()

    # Sortieren für Lesbarkeit
    season = season.sort_values(["year", "driver_name"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    season.to_csv(output_path, index=False)
    print(f"✅ F1 season features written to: {output_path}")

    return output_path


if __name__ == "__main__":
    build_f1_season_features()
