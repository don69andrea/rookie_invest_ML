from pathlib import Path
import pandas as pd
import numpy as np


INTERIM_DIR = Path("data/f3/interim")
PROCESSED_DIR = Path("data/f3/processed")


def build_f3_season_features_advanced(
    input_path: str | Path = PROCESSED_DIR / "f3_2019_2025_races_features.csv",
    output_path: str | Path = PROCESSED_DIR / "f3_features_advanced.csv",
) -> Path:
    """
    Erstellt ein erweitertes Season Feature Set für F3 auf Basis
    des bestehenden Race Feature Datensatzes.

    Erwarteter Input: dein altes Rennen Features CSV mit Spalten wie
    season, race_id, driver_name, driver_code, team_name, laps, kph,
    position_clean, avg_lap_time_s, time_from_winner_s,
    best_lap_from_best_s, driver_speed, team_speed,
    team_avg_pos_season, driver_vs_team, lap_vs_race_avg,
    is_dnf, is_dns, is_dsq.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Advanced F3 race features file not found: {input_path}"
        )

    df = pd.read_csv(input_path, low_memory=False)

    # numerische Typen
    num_cols = [
        "season",
        "race_id",
        "laps",
        "kph",
        "position_clean",
        "avg_lap_time_s",
        "time_from_winner_s",
        "best_lap_from_best_s",
        "race_avg_lap_time_s",
        "driver_speed",
        "team_speed",
        "team_avg_pos_season",
        "driver_vs_team",
        "lap_vs_race_avg",
        "is_dnf",
        "is_dns",
        "is_dsq",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strings bereinigen
    for col in ["driver_name", "driver_code", "team_name", "status", "session_type"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Fahrer Saison Aggregation
    group_cols = ["season", "driver_name", "driver_code", "team_name"]
    group = df.groupby(group_cols)

    agg = group.agg(
        n_races=("race_id", "nunique"),

        # Finishing Performance
        avg_finish=("position_clean", "mean"),
        best_finish=("position_clean", "min"),
        worst_finish=("position_clean", "max"),
        finish_std=("position_clean", "std"),
        wins=("position_clean", lambda s: (s == 1).sum()),
        podiums=("position_clean", lambda s: (s <= 3).sum()),
        top10_finishes=("position_clean", lambda s: (s <= 10).sum()),

        # Ausfälle
        dnf_count=("is_dnf", "sum"),
        dnf_rate=("is_dnf", "mean"),
        dns_count=("is_dns", "sum"),
        dns_rate=("is_dns", "mean"),
        dsq_count=("is_dsq", "sum"),
        dsq_rate=("is_dsq", "mean"),

        # Runden und Speed
        total_laps=("laps", "sum"),
        avg_kph=("kph", "mean"),
        avg_lap_time_s=("avg_lap_time_s", "mean"),
        avg_time_from_winner_s=("time_from_winner_s", "mean"),
        avg_best_lap_from_best_s=("best_lap_from_best_s", "mean"),

        # Pace relativ zu Team und Feld
        driver_speed_mean=("driver_speed", "mean"),
        team_speed_mean=("team_speed", "mean"),
        team_avg_pos_season_mean=("team_avg_pos_season", "mean"),

        driver_vs_team_mean=("driver_vs_team", "mean"),
        driver_vs_team_best=("driver_vs_team", "min"),
        driver_vs_team_std=("driver_vs_team", "std"),

        lap_vs_race_avg_mean=("lap_vs_race_avg", "mean"),
        lap_vs_race_avg_std=("lap_vs_race_avg", "std"),
    ).reset_index()

    # Raten
    agg["win_rate"] = agg["wins"] / agg["n_races"]
    agg["podium_rate"] = agg["podiums"] / agg["n_races"]
    agg["top10_rate"] = agg["top10_finishes"] / agg["n_races"]

    # Serienlabel
    agg["series"] = "F3"

    # Spaltenreihenfolge
    col_order = [
        "series",
        "season",
        "driver_name",
        "driver_code",
        "team_name",
        "n_races",
        "avg_finish",
        "best_finish",
        "worst_finish",
        "finish_std",
        "wins",
        "win_rate",
        "podiums",
        "podium_rate",
        "top10_finishes",
        "top10_rate",
        "dnf_count",
        "dnf_rate",
        "dns_count",
        "dns_rate",
        "dsq_count",
        "dsq_rate",
        "total_laps",
        "avg_kph",
        "avg_lap_time_s",
        "avg_time_from_winner_s",
        "avg_best_lap_from_best_s",
        "driver_speed_mean",
        "team_speed_mean",
        "team_avg_pos_season_mean",
        "driver_vs_team_mean",
        "driver_vs_team_best",
        "driver_vs_team_std",
        "lap_vs_race_avg_mean",
        "lap_vs_race_avg_std",
    ]

    # falls einzelne Spalten fehlen, crasht es nicht gleich
    col_order = [c for c in col_order if c in agg.columns]
    agg = agg[col_order].copy()

    # sortieren für Lesbarkeit
    agg = agg.sort_values(["season", "driver_name"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)
    print(f"F3 advanced season features written to: {output_path}")

    return output_path


if __name__ == "__main__":
    build_f3_season_features_advanced()
