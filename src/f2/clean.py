from pathlib import Path
import pandas as pd


INTERIM_DIR = Path("data/f2/interim")
PROCESSED_DIR = Path("data/f2/processed")


def load_f2_sessions_raw(path: str | Path = INTERIM_DIR / "f2_sessions_raw.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"F2 sessions raw file not found: {path}")
    return pd.read_csv(path)


def load_f2_results_raw(path: str | Path = INTERIM_DIR / "f2_race_results_raw.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"F2 race results raw file not found: {path}")
    return pd.read_csv(path)


def clean_f2_sessions(
    input_path: str | Path = INTERIM_DIR / "f2_sessions_raw.csv",
    output_path: str | Path = INTERIM_DIR / "f2_sessions_clean.csv",
) -> Path:
    """
    Bereinigt die F2-Session-Daten:
    - Trim von Strings
    - Typen für round, position, laps, kph
    - Datum in datetime
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_f2_sessions_raw(input_path)

    # Strings trimmen (driver, team, circuit)
    for col in ["driver_name", "team_name", "circuit_name", "session_type", "session_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Runde als int
    if "round" in df.columns:
        # round ist aktuell z.B. "Round 1" → Zahl extrahieren
        df["round_stripped"] = df["round"].astype(str).str.extract(r"(\d+)", expand=False)
        df["round"] = pd.to_numeric(df["round_stripped"], errors="coerce").astype("Int64")
        df.drop(columns=["round_stripped"], inplace=True)

    # Datum
    if "session_date" in df.columns:
        df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce")

    # Numerische Spalten
    for col in ["position", "laps", "car_number", "kph", "best_lap_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional: sortieren
    sort_cols = [c for c in ["session_date", "round", "session_category", "driver_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"✅ F2 sessions CLEAN table written to: {output_path}")
    return output_path


def clean_f2_results(
    input_path: str | Path = INTERIM_DIR / "f2_race_results_raw.csv",
    output_path: str | Path = INTERIM_DIR / "f2_race_results_clean.csv",
) -> Path:
    """
    Bereinigt die F2-Rennergebnisse:
    - Spalten umbenennen (Lowercase mit Unterstrich)
    - Datentypen
    - Datum
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_f2_results_raw(input_path)

    # Spalten umbenennen in einheitliches Schema
    col_map = {
        "Season": "season",
        "Round": "round",
        "Track Name": "circuit_name",
        "Country": "country",
        "City": "city",
        "Date": "race_date",
        "Length (Km)": "track_length_km",
        "Race Type": "race_type",
        "Position": "position",
        "Car Number": "car_number",
        "Driver Name": "driver_name",
        "Team Name": "team_name",
        "Laps": "laps",
        "Time": "time",
        "Gap": "gap",
        "Interval": "interval",
        "KPH": "kph",
        "Best Lap Time": "best_lap_time",
    }

    existing_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=existing_map)

    # Strings trimmen
    for col in ["driver_name", "team_name", "circuit_name", "race_type", "country", "city"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Typen setzen
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")

    for col in ["position", "laps", "car_number", "kph"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "race_date" in df.columns:
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # Sortierung
    sort_cols = [c for c in ["season", "round", "race_type", "driver_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"✅ F2 race results CLEAN table written to: {output_path}")
    return output_path


def clean_all_f2() -> None:
    clean_f2_sessions()
    clean_f2_results()


if __name__ == "__main__":
    clean_all_f2()
