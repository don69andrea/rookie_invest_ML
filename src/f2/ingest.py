from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/f2/raw")
INTERIM_DIR = Path("data/f2/interim")


SESSION_FILES = {
    "Feature-Race.csv": "FEATURE_RACE",
    "Sprint-Race.csv": "SPRINT_RACE_1",
    "Sprint-Race-2.csv": "SPRINT_RACE_2",
    "Qualifying-Session.csv": "QUALIFYING",
    "Free-Practice.csv": "FREE_PRACTICE",
}


RESULTS_FILE = "Formula2_Race_Results.csv"


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"⚠️  File not found (skipped): {path}")
        return None
    return pd.read_csv(path)


def load_f2_session_tables(raw_dir: Path = RAW_DIR) -> dict:
    """
    Lädt alle F2-Session-Files (Feature, Sprint, Quali, FP) aus data/f2/raw.
    Gibt ein Dict {filename: DataFrame} zurück (nur vorhandene Files).
    """
    raw_dir = Path(raw_dir)
    tables: dict[str, pd.DataFrame] = {}

    for filename, session_category in SESSION_FILES.items():
        path = raw_dir / filename
        df = _read_csv_if_exists(path)
        if df is not None:
            tables[filename] = df
    return tables


def normalize_session_df(df: pd.DataFrame, session_category: str, source_file: str) -> pd.DataFrame:
    """
    Vereinheitlicht die Spaltennamen einer F2-Session-Tabelle und fügt Metadaten-Spalten hinzu.
    Erwartete Originalspalten (Großbuchstaben, wie in deinen Files):
        'LAPS', 'TIME', 'GAP', 'INT.', 'KPH', 'BEST', 'LAP',
        'POS', 'CAR', 'PILOT NAME', 'TEAM', 'CIRCUIT', 'TYPE', 'ROUND', 'DATE'
    """

    col_map = {
        "LAPS": "laps",
        "TIME": "time",
        "GAP": "gap",
        "INT.": "interval",
        "KPH": "kph",
        "BEST": "best_lap",
        "LAP": "best_lap_number",
        "POS": "position",
        "CAR": "car_number",
        "PILOT NAME": "driver_name",
        "TEAM": "team_name",
        "CIRCUIT": "circuit_name",
        "TYPE": "session_type",
        "ROUND": "round",
        "DATE": "session_date",
    }

    # Nur Spalten umbenennen, die es auch wirklich gibt
    existing_map = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=existing_map).copy()

    # Metadaten hinzufügen
    df["session_category"] = session_category
    df["source_file"] = source_file
    df["series"] = "F2"

    # ein paar sinnvolle Spalten vorne anordnen
    preferred_order = [
        "series",
        "session_category",
        "session_type",
        "round",
        "session_date",
        "driver_name",
        "car_number",
        "team_name",
        "position",
        "laps",
        "time",
        "gap",
        "interval",
        "kph",
        "best_lap",
        "best_lap_number",
        "circuit_name",
        "source_file",
    ]

    cols = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    df = df[cols]

    return df


def build_f2_sessions_raw(
    raw_dir: str | Path = RAW_DIR,
    output_path: str | Path = INTERIM_DIR / "f2_sessions_raw.csv",
) -> Path:
    """
    Liest alle Session-Files (Feature, Sprint, Quali, Free Practice) ein,
    vereinheitlicht die Struktur und schreibt eine gemeinsame Tabelle:
        data/f2/interim/f2_sessions_raw.csv
    """

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tables = load_f2_session_tables(raw_dir)

    if not tables:
        raise FileNotFoundError(f"Keine F2-Session-Files in {raw_dir} gefunden.")

    normalized_dfs: list[pd.DataFrame] = []
    for filename, session_category in SESSION_FILES.items():
        if filename not in tables:
            continue
        df = tables[filename]
        norm_df = normalize_session_df(df, session_category=session_category, source_file=filename)
        normalized_dfs.append(norm_df)

    if not normalized_dfs:
        raise RuntimeError("Keine F2-Session-DataFrames konnten normalisiert werden.")

    sessions_all = pd.concat(normalized_dfs, ignore_index=True)

    # Optional: sortieren nach Date + Session-Kategorie + Fahrername
    sort_cols = [c for c in ["session_date", "round", "session_category", "driver_name"] if c in sessions_all.columns]
    if sort_cols:
        sessions_all = sessions_all.sort_values(sort_cols).reset_index(drop=True)

    sessions_all.to_csv(output_path, index=False)
    print(f"✅ F2 sessions raw table written to: {output_path}")

    return output_path


def build_f2_results_raw(
    raw_dir: str | Path = RAW_DIR,
    output_path: str | Path = INTERIM_DIR / "f2_race_results_raw.csv",
) -> Path | None:
    """
    Kopiert die Formel-2-Rennergebnisse nach interim:
        data/f2/interim/f2_race_results_raw.csv

    Noch ohne starke Umbenennung, damit wir in einem späteren Schritt
    (clean.py) auf Basis der realen Spalten entscheiden, wie wir joinen.
    """

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src = raw_dir / RESULTS_FILE
    if not src.exists():
        print(f"⚠️  F2 results file not found: {src} (skipping).")
        return None

    df_results = pd.read_csv(src)
    df_results.to_csv(output_path, index=False)
    print(f"✅ F2 race results raw table written to: {output_path}")

    return output_path


def ingest_all_f2(
    raw_dir: str | Path = RAW_DIR,
    sessions_output: str | Path = INTERIM_DIR / "f2_sessions_raw.csv",
    results_output: str | Path = INTERIM_DIR / "f2_race_results_raw.csv",
) -> None:
    """
    Convenience-Funktion: baut sowohl die Sessions- als auch die Results-Rohbasis.
    """

    build_f2_sessions_raw(raw_dir=raw_dir, output_path=sessions_output)
    build_f2_results_raw(raw_dir=raw_dir, output_path=results_output)


if __name__ == "__main__":
    ingest_all_f2()
