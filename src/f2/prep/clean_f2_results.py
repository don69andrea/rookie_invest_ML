import pandas as pd
import re
from pathlib import Path

# Pfade anpassen falls nötig
INPUT = Path("data/f2/raw/f2_results_fia.csv")
OUTPUT = Path("data/f2/processed/f2_results_fia_clean.csv")


def parse_driver_info(cell):
    """
    Zerlegt die Fahrer-Spalte in:
    status (DNF, DNS usw), car_number, driver_name, driver_code, team_name
    """
    if pd.isna(cell):
        return pd.Series(
            [None, None, None, None, None],
            index=["status", "car_number", "driver_name", "driver_code", "team_name"],
        )

    text = str(cell).strip().replace("\xa0", " ")

    # Optionaler Status am Anfang (z. B. DNF, DNS, DSQ), dann Startnummer
    m = re.match(r"^(?P<status>[A-Z]+)?(?P<number>\d+)\s*(?P<rest>.*)$", text)
    if not m:
        return pd.Series(
            [None, None, text, None, None],
            index=["status", "car_number", "driver_name", "driver_code", "team_name"],
        )

    status = m.group("status")
    car_number = m.group("number")
    rest = m.group("rest").strip()

    # Ersten Dreierblock Grossbuchstaben als Driver Code
    m2 = re.search(r"([A-Z]{3})", rest)
    if not m2:
        return pd.Series(
            [status, car_number, rest, None, None],
            index=["status", "car_number", "driver_name", "driver_code", "team_name"],
        )

    driver_code = m2.group(1)
    name_part = rest[: m2.start()].strip()
    team_part = rest[m2.end() :].strip()

    # Punkt nach Initialen entfernen
    name_part = name_part.replace(".", " ").strip()

    return pd.Series(
        [status, car_number, name_part, driver_code, team_part],
        index=["status", "car_number", "driver_name", "driver_code", "team_name"],
    )


def main():
    print(f"Lade Rohdaten aus {INPUT}")
    df = pd.read_csv(INPUT)

    # Spaltennamen bereinigen
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r"[^\w\s]+", "_", regex=True)

    # Wichtige Umbenennungen
    df = df.rename(
        columns={
            "POSNumber _ Driver and TeamNo _ Driver": "driver_info",
            "LAP SET ON": "lap_set_on",
        }
    )

    # Fahrerinfo parsen
    parsed = df["driver_info"].apply(parse_driver_info)

    # Alte Spalten entfernen falls vorhanden und neue anhängen
    df = pd.concat(
        [
            df.drop(
                columns=["status", "car_number", "driver_name", "driver_code", "team_name"],
                errors="ignore",
            ),
            parsed,
        ],
        axis=1,
    )

    # ein bisschen Typen aufräumen, ohne zu aggressiv zu casten
    for col in ["season", "round", "race_id", "laps", "best_lap_number"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["kph"] = pd.to_numeric(df["kph"], errors="coerce")

    # optional sortieren
    df = df.sort_values(["season", "round", "session", "race_id"]).reset_index(drop=True)

    df.to_csv(OUTPUT, index=False)
    print(f"Fertig. Gespeichert unter {OUTPUT}")


if __name__ == "__main__":
    main()
