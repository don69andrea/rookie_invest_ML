from pathlib import Path
import pandas as pd


# Standardpfade (relativ zum Projekt-Root)
RAW_DIR = Path("data/f1/raw")
INTERIM_DIR = Path("data/f1/interim")


def load_f1_raw_tables(raw_dir: Path = RAW_DIR) -> dict:
    """
    Lädt alle relevanten F1-Roh-CSV-Dateien und gibt sie als Dict zurück.
    Erwartet das klassische F1-Kaggle-Schema.
    """
    raw_dir = Path(raw_dir)

    tables = {}

    def read_csv(name: str) -> pd.DataFrame:
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Expected raw file not found: {path}")
        return pd.read_csv(path)

    tables["races"] = read_csv("races")
    tables["results"] = read_csv("results")
    tables["drivers"] = read_csv("drivers")
    tables["constructors"] = read_csv("constructors")
    tables["circuits"] = read_csv("circuits")
    tables["status"] = read_csv("status")

    # Optional – können wir später für Features nutzen
    for optional in [
        "lap_times",
        "pit_stops",
        "qualifying",
        "sprint_results",
        "driver_standings",
        "constructor_standings",
        "constructor_results",
        "seasons",
    ]:
        path = raw_dir / f"{optional}.csv"
        if path.exists():
            tables[optional] = pd.read_csv(path)

    return tables


def build_f1_race_driver_raw(
    raw_dir: str | Path = RAW_DIR,
    output_path: str | Path = INTERIM_DIR / "f1_race_driver_raw.csv",
) -> Path:
    """
    Baut eine grundlegende "one row per driver per race"-Tabelle,
    in der die wichtigsten Infos aus races, results, drivers, constructors,
    circuits und status zusammengeführt sind.

    Noch KEIN Feature Engineering, nur Joins + saubere Spaltennamen.
    """

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tables = load_f1_raw_tables(raw_dir)

    races = tables["races"].copy()
    results = tables["results"].copy()
    drivers = tables["drivers"].copy()
    constructors = tables["constructors"].copy()
    circuits = tables["circuits"].copy()
    status = tables["status"].copy()

    # --- Spalten umbenennen, damit beim Mergen nichts kollidiert ---

    # races
    races = races.rename(
        columns={
            "raceId": "race_id",
            "year": "year",
            "round": "round",
            "circuitId": "circuit_id",
            "name": "race_name",
            "date": "race_date",
            "time": "race_time",
            "url": "race_url",
        }
    )

    # results
    results = results.rename(
        columns={
            "resultId": "result_id",
            "raceId": "race_id",
            "driverId": "driver_id",
            "constructorId": "constructor_id",
            "grid": "grid_position",
            "position": "finishing_position",
            "positionText": "finishing_position_text",
            "positionOrder": "finishing_order",
            "points": "points",
            "laps": "laps_completed",
            "time": "result_time",
            "milliseconds": "result_ms",
            "fastestLap": "fastest_lap_number",
            "rank": "fastest_lap_rank",
            "fastestLapTime": "fastest_lap_time",
            "fastestLapSpeed": "fastest_lap_speed",
            "statusId": "status_id",
        }
    )

    # drivers
    drivers = drivers.rename(
        columns={
            "driverId": "driver_id",
            "driverRef": "driver_ref",
            "number": "driver_number",
            "code": "driver_code",
            "forename": "forename",
            "surname": "surname",
            "dob": "driver_dob",
            "nationality": "driver_nationality",
            "url": "driver_url",
        }
    )

    # constructors
    constructors = constructors.rename(
        columns={
            "constructorId": "constructor_id",
            "constructorRef": "constructor_ref",
            "name": "constructor_name",
            "nationality": "constructor_nationality",
            "url": "constructor_url",
        }
    )

    # circuits
    circuits = circuits.rename(
        columns={
            "circuitId": "circuit_id",
            "circuitRef": "circuit_ref",
            "name": "circuit_name",
            "location": "circuit_location",
            "country": "circuit_country",
            "lat": "circuit_lat",
            "lng": "circuit_lng",
            "alt": "circuit_alt",
            "url": "circuit_url",
        }
    )

    # status
    status = status.rename(
        columns={
            "statusId": "status_id",
            "status": "status_text",
        }
    )

    # --- Merges: results + races + drivers + constructors + circuits + status ---

    df = results.merge(races, on="race_id", how="left")
    df = df.merge(drivers, on="driver_id", how="left")
    df = df.merge(constructors, on="constructor_id", how="left")
    df = df.merge(circuits, on="circuit_id", how="left")
    df = df.merge(status, on="status_id", how="left")

    # Optional: eine schönere Driver-Name-Spalte
    df["driver_name"] = (df["forename"].fillna("") + " " + df["surname"].fillna("")).str.strip()

    # Ein bisschen sortieren für bessere Lesbarkeit
    df = df.sort_values(["year", "round", "race_id", "finishing_order"]).reset_index(drop=True)

    # Speichern
    df.to_csv(output_path, index=False)
    print(f"✅ F1 race-driver raw table written to: {output_path}")

    return output_path


if __name__ == "__main__":
    build_f1_race_driver_raw()
