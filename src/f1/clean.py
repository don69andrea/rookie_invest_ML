from pathlib import Path
import pandas as pd


INTERIM_DIR = Path("data/f1/interim")
PROCESSED_DIR = Path("data/f1/processed")


def load_race_driver_raw(path: str | Path = INTERIM_DIR / "f1_race_driver_raw.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw race-driver file not found: {path}")
    # low_memory=False, damit Pandas die Typen besser erkennen kann
    return pd.read_csv(path, low_memory=False)


def clean_f1_race_driver(
    input_path: str | Path = INTERIM_DIR / "f1_race_driver_raw.csv",
    output_path: str | Path = INTERIM_DIR / "f1_race_driver_clean.csv",
) -> Path:
    """
    Leichtes Cleaning für F1:
    - Typen setzen
    - Helferspalten erstellen (DNF, Classified, Points-Finish)
    - Optionale Filter (z.B. nach Jahr) vorbereiten
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_race_driver_raw(input_path)

    # --- Typen setzen / konvertieren ---

    # Jahreszahl & Runde
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")

    # Punkte, Grid, Position
    for col in ["grid_position", "finishing_position", "finishing_order", "laps_completed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "points" in df.columns:
        df["points"] = pd.to_numeric(df["points"], errors="coerce")

    # Zeiten in Millisekunden
    if "result_ms" in df.columns:
        df["result_ms"] = pd.to_numeric(df["result_ms"], errors="coerce")

    # Race-Datum
    if "race_date" in df.columns:
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # --- Helper-Flags: DNF, Classified, Points-Finish ---

    # status_text kann z. B. 'Finished', '+1 Lap', 'Accident', 'Engine', ...
    status_col = "status_text"
    if status_col in df.columns:
        finished_like = ["Finished", "Finished (e.g.)"]
        # classified, wenn kein klassischer DNF-Status
        df["is_classified_finish"] = df[status_col].str.contains("Finished", na=False)
        df["is_dnf"] = ~df["is_classified_finish"]
    else:
        df["is_classified_finish"] = pd.NA
        df["is_dnf"] = pd.NA

    # Punkte-Finish
    if "points" in df.columns:
        df["is_points_finish"] = df["points"].fillna(0) > 0
    else:
        df["is_points_finish"] = pd.NA

    # --- Optional: sehr alte Jahre rausfiltern (z.B. ab 1980/1990) ---
    # Noch nicht aktiviert – nur vorbereitet:
    # df = df[df["year"] >= 1990].copy()

    # Sortierung
    sort_cols = [c for c in ["year", "round", "race_id", "finishing_order"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Speichern
    df.to_csv(output_path, index=False)
    print(f"✅ F1 race-driver CLEAN table written to: {output_path}")

    return output_path


if __name__ == "__main__":
    clean_f1_race_driver()
