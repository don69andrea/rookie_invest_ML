from __future__ import annotations

from pathlib import Path
import pandas as pd


RACE_PATH = Path("data/f3/processed/f3_2019_2025_races_features.csv")
ADV_PATH = Path("data/f3/processed/f3_features_advanced.csv")


def main(sample_size: int = 5, tolerance: float | None = None) -> None:
    if not RACE_PATH.exists():
        raise FileNotFoundError(f"Missing race features file: {RACE_PATH}")
    if not ADV_PATH.exists():
        raise FileNotFoundError(f"Missing advanced features file: {ADV_PATH}")

    race = pd.read_csv(RACE_PATH, low_memory=False)
    adv = pd.read_csv(ADV_PATH, low_memory=False)

    required_race = {
        "season",
        "driver_name",
        "driver_code",
        "team_name",
        "position_clean",
        "laps",
        "kph",
        "avg_lap_time_s",
        "time_from_winner_s",
        "best_lap_from_best_s",
        "driver_speed",
        "team_speed",
        "team_avg_pos_season",
        "driver_vs_team",
        "lap_vs_race_avg",
        "is_dnf",
        "is_dns",
        "is_dsq",
    }
    missing = required_race.difference(race.columns)
    if missing:
        raise ValueError(f"Missing columns in race features: {sorted(missing)}")

    # Recompute advanced aggregates from race features
    group_cols = ["season", "driver_name", "driver_code", "team_name"]
    recomputed = (
        race.groupby(group_cols)
        .agg(
            n_races=("race_id", "nunique"),
            avg_finish=("position_clean", "mean"),
            best_finish=("position_clean", "min"),
            worst_finish=("position_clean", "max"),
            finish_std=("position_clean", "std"),
            wins=("position_clean", lambda s: (s == 1).sum()),
            podiums=("position_clean", lambda s: (s <= 3).sum()),
            top10_finishes=("position_clean", lambda s: (s <= 10).sum()),
            dnf_count=("is_dnf", "sum"),
            dnf_rate=("is_dnf", "mean"),
            dns_count=("is_dns", "sum"),
            dns_rate=("is_dns", "mean"),
            dsq_count=("is_dsq", "sum"),
            dsq_rate=("is_dsq", "mean"),
            total_laps=("laps", "sum"),
            avg_kph=("kph", "mean"),
            avg_lap_time_s=("avg_lap_time_s", "mean"),
            avg_time_from_winner_s=("time_from_winner_s", "mean"),
            avg_best_lap_from_best_s=("best_lap_from_best_s", "mean"),
            driver_speed_mean=("driver_speed", "mean"),
            team_speed_mean=("team_speed", "mean"),
            team_avg_pos_season_mean=("team_avg_pos_season", "mean"),
            driver_vs_team_mean=("driver_vs_team", "mean"),
            driver_vs_team_best=("driver_vs_team", "min"),
            driver_vs_team_std=("driver_vs_team", "std"),
            lap_vs_race_avg_mean=("lap_vs_race_avg", "mean"),
            lap_vs_race_avg_std=("lap_vs_race_avg", "std"),
        )
        .reset_index()
    )

    recomputed["win_rate"] = recomputed["wins"] / recomputed["n_races"]
    recomputed["podium_rate"] = recomputed["podiums"] / recomputed["n_races"]
    recomputed["top10_rate"] = recomputed["top10_finishes"] / recomputed["n_races"]

    # Join and compare
    merged = recomputed.merge(
        adv,
        on=["season", "driver_name", "driver_code", "team_name"],
        suffixes=("_recomputed", "_adv"),
        how="inner",
    )

    if merged.empty:
        raise RuntimeError("No matching rows between race features and advanced features.")

    compare_cols = [
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

    diffs = {}
    for col in compare_cols:
        a = merged[f"{col}_recomputed"]
        b = merged[f"{col}_adv"]
        diffs[col] = (a - b).abs().max()

    sample = merged.sample(min(sample_size, len(merged)), random_state=42)

    print("Max absolute diffs (recomputed vs advanced):")
    for col, val in sorted(diffs.items(), key=lambda kv: kv[1], reverse=True):
        print(f"- {col}: {val}")

    print("\nSample rows:")
    print(sample[["season", "driver_name", "driver_code", "team_name"] + [f"{c}_adv" for c in compare_cols[:5]]])

    if tolerance is not None:
        worst = max(diffs.values()) if diffs else 0.0
        if worst > tolerance:
            raise AssertionError(f"Max diff {worst} exceeds tolerance {tolerance}")


if __name__ == "__main__":
    main()
