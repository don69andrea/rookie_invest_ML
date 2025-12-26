import time
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
import requests


BASE_URL = "https://www.fiaformula2.com/Results?raceid={race_id}"
MANUAL_META_PATH = Path("data/f2/raw/f2_dataset_manuell.xlsx")
OUTPUT_PATH = Path("data/f2/raw/f2_results_fia.csv")

# Falls du weniger aggressiv scrapen willst, hier erhöhen
REQUEST_SLEEP_SECONDS = 1.0

def finde_klassifikationstabelle(tables: list[pd.DataFrame], race_id: int) -> pd.DataFrame:
    """
    Wählt die Ergebnis Klassifikationstabelle aus den auf der FIA Seite
    gefundenen Tabellen aus.

    Die FIA Seite hat je nach Jahr und Layout verschiedene Spaltennamen.
    Wir sind deshalb absichtlich tolerant und prüfen nur, ob typische
    Spalten für eine Rennergebnis Tabelle vorkommen.
    """
    candidates: list[pd.DataFrame] = []
    header_strings: list[str] = []

    for df in tables:
        # Spaltennamen zu einem String verbinden, nur zur Diagnose
        header_str = " | ".join(str(c) for c in df.columns)
        header_strings.append(header_str)

        h_up = header_str.upper()

        # sehr einfache, robuste Bedingung
        # wir erwarten Spalten zu Position, Runden und Zeit
        if "POS" in h_up and "LAPS" in h_up and "TIME" in h_up:
            candidates.append(df)

    if not candidates:
        # Diese Fehlermeldung siehst du ja bereits in deinem Log
        raise RuntimeError(
            f"Keine Klassifikationstabelle fuer race_id={race_id}. "
            f"Gefundene Tabellen Header: {header_strings}"
        )

    # Falls mehrere Tabellen passen, nimm die mit den meisten Zeilen
    candidates.sort(key=lambda d: len(d), reverse=True)
    klass_df = candidates[0]

    # Optional: Spaltennamen ein wenig aufräumen
    # Hier kannst du später noch Mapping machen, z B:
    # rename_map = {
    #     "POSNumber / Driver and TeamNo / Driver": "POS_DRIVER",
    #     ...
    # }
    # klass_df = klass_df.rename(columns=rename_map)

    return klass_df

def get_http_session() -> requests.Session:
    """
    Erzeugt eine Requests Session mit einem halbwegs realistischen User Agent.
    Eine Session ist effizienter als für jeden Request ein neues Objekt anzulegen.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
    )
    return session


def load_manual_race_list(path: Path) -> pd.DataFrame:
    """
    Lädt die manuelle F2 Metadatei.
    Erwartet mindestens die Spalten:
        - season
        - race_id

    Wenn keine Spalte round vorhanden ist, wird sie automatisch
    innerhalb jeder Saison als laufende Nummer generiert.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Metadatei nicht gefunden: {path}. "
            f"Erwarte eine Excel Datei mit mindestens den Spalten 'season' und 'race_id'."
        )

    df = pd.read_excel(path)

    required_cols = {"season", "race_id"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Metadatei {path} fehlt folgende Spalten: {', '.join(sorted(missing))}"
        )

    # Saison und race_id als int säubern
    df["season"] = df["season"].astype(int)
    df["race_id"] = df["race_id"].astype(int)

    # Wenn round fehlt, automatisch pro Saison generieren
    if "round" not in df.columns:
        df = (
            df.sort_values(["season", "race_id"])
            .reset_index(drop=True)
        )
        df["round"] = (
            df.groupby("season").cumcount() + 1
        )

    # Zur Sicherheit nach season, round sortieren
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    return df


def fetch_race_html(session: requests.Session, race_id: int) -> str:
    """
    Lädt die HTML Seite für ein gegebenes race_id von der FIA F2 Seite.
    """
    url = BASE_URL.format(race_id=race_id)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text

def extract_classification_tables_from_html(html: str, race_id: int) -> list[pd.DataFrame]:
    """
    Liest alle Tabellen aus dem HTML und delegiert die Auswahl
    der Klassifikationstabelle an 'finde_klassifikationstabelle'.
    """

    # Tabellen extrahieren (robust dank StringIO)
    tables = pd.read_html(StringIO(html))

    # Nur die eine Klassifikationstabelle auswählen
    klass_df = finde_klassifikationstabelle(tables, race_id)

    # In Liste zurückgeben, weil parse_fia_race mehrere Sessions erwartet
    return [klass_df]

def normalize_result_table(
    df: pd.DataFrame,
    season: int,
    round_number: int,
    race_id: int,
    session_label: str,
) -> pd.DataFrame:
    """
    Bereitet eine FIA F2 Ergebnistabelle minimal auf und ergänzt Metadaten.

    Ziel:
        - Spaltennamen normalisieren
        - Metafelder hinzufügen
        - Tabellenstruktur nicht zu stark annehmen,
          da die FIA Seite sonst schnell Änderungen verursachen kann.

    Wir erwarten nach aktueller FIA Struktur z.B. Spalten:
        POS
        Number / Driver and Team No / Driver
        LAPS
        TIME
        GAP
        INT.
        KPH
        BEST
        LAP
    """

    df = df.copy()

    # Originale Spaltennamen für dokumentierte Nachvollziehbarkeit speichern
    df.attrs["original_columns"] = list(df.columns)

    # Spaltennamen normalisieren
    rename_map = {}
    for col in df.columns:
        col_str = str(col).strip()

        lower = col_str.lower()
        if lower == "pos":
            rename_map[col] = "position"
        elif lower.startswith("number / driver"):
            rename_map[col] = "number_driver_team"
        elif lower == "laps":
            rename_map[col] = "laps"
        elif lower == "time":
            rename_map[col] = "race_time"
        elif lower == "gap":
            rename_map[col] = "gap"
        elif "int" in lower:
            rename_map[col] = "interval"
        elif lower == "kph":
            rename_map[col] = "kph"
        elif lower == "best":
            rename_map[col] = "best_lap_time"
        elif lower == "lap":
            rename_map[col] = "best_lap_number"
        # alles andere unverändert lassen

    df = df.rename(columns=rename_map)

    # Metadaten ergänzen
    df["season"] = season
    df["round"] = round_number
    df["race_id"] = race_id
    df["session"] = session_label  # z.B. "Sprint Race" oder "Feature Race"

    # Reihenfolge grob ordnen, falls die Spalten vorhanden sind
    preferred_order = [
        "season",
        "round",
        "race_id",
        "session",
        "position",
        "number_driver_team",
        "laps",
        "race_time",
        "gap",
        "interval",
        "kph",
        "best_lap_time",
        "best_lap_number",
    ]

    existing_cols = [c for c in preferred_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    return df


def parse_fia_race(
    session: requests.Session,
    season: int,
    round_number: int,
    race_id: int,
) -> pd.DataFrame:
    """
    Lädt die FIA Ergebnisseite für ein Rennen und gibt ein DataFrame
    mit allen gefundenen Klassifikationstabellen zurück.

    Typischerweise ergibt das zwei Tabellen:
        - Sprint Race Results
        - Feature Race Results

    Wenn der Lauf nicht gefunden oder anders strukturiert ist,
    wird eine Exception geworfen.
    """
    html = fetch_race_html(session, race_id)
    tables = extract_classification_tables_from_html(html, race_id)

    # Wir gehen davon aus, dass in der Reihenfolge zuerst Sprint, dann Feature kommt.
    # Das ist eine Annahme, die man später verfeinern kann.
    session_labels = []
    if len(tables) == 1:
        session_labels = ["Race"]
    elif len(tables) == 2:
        session_labels = ["Sprint Race", "Feature Race"]
    else:
        # Mehr als zwei Klassifikationstabellen. Wir labeln generisch.
        session_labels = [f"Race {i + 1}" for i in range(len(tables))]

    result_frames: List[pd.DataFrame] = []
    for tbl, label in zip(tables, session_labels):
        cleaned = normalize_result_table(
            tbl,
            season=season,
            round_number=round_number,
            race_id=race_id,
            session_label=label,
        )
        result_frames.append(cleaned)

    combined = pd.concat(result_frames, ignore_index=True)
    return combined


def ingest_all_races() -> None:
    """
    Top Level Routine:
        - Metadatei laden
        - Alle Rennen iterieren
        - Ergebnisse in eine kombinierte CSV Datei schreiben
    """
    meta = load_manual_race_list(MANUAL_META_PATH)

    session = get_http_session()

    all_results: List[pd.DataFrame] = []
    success_count = 0
    fail_count = 0

    for _, row in meta.iterrows():
        season = int(row["season"])
        round_number = int(row["round"])
        race_id = int(row["race_id"])

        url = BASE_URL.format(race_id=race_id)
        print(
            f"⏳ Hole {season} round {round_number} "
            f"race_id={race_id} ({url})"
        )

        try:
            df_race = parse_fia_race(
                session=session,
                season=season,
                round_number=round_number,
                race_id=race_id,
            )
            all_results.append(df_race)
            success_count += 1
            print(
                f"✅ Erfolgreich: season={season}, round={round_number}, "
                f"race_id={race_id} mit {len(df_race)} Zeilen"
            )
        except Exception as exc:
            fail_count += 1
            print(
                f"❌ Fehler bei season={season}, round={round_number}, "
                f"race_id={race_id}: {exc}"
            )

        time.sleep(REQUEST_SLEEP_SECONDS)

    if not all_results:
        raise RuntimeError("Keine Rennen erfolgreich geladen")

    combined = pd.concat(all_results, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print()
    print(f"Fertig. Erfolgreich geladene Rennen: {success_count}")
    print(f"Fehlgeschlagene Rennen: {fail_count}")
    print(f"Gespeichert unter: {OUTPUT_PATH}")


if __name__ == "__main__":
    ingest_all_races()
