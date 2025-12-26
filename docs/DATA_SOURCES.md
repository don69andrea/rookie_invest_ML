# Data Sources

Diese Datei fasst die verwendeten Datenquellen zusammen und beschreibt, wo sie
im Projekt abgelegt werden. So bleibt nachvollziehbar, welche Rohdaten in die
Pipeline einfliessen und welche Outputs daraus entstehen.

## F1 (Kaggle Schema)
- Quelle: F1 Kaggle Datensatz (klassisches Schema)
- Pfad: `data/f1/raw/`
- Erwartete Dateien: races.csv, results.csv, drivers.csv,
  constructors.csv, circuits.csv, status.csv (optional weitere)

## F2 (FIA Ergebnisse)
- Quelle: FIA Formula 2 Results (Scraper)
- Pfad Meta: `data/f2/raw/f2_dataset_manuell.xlsx`
- Ergebnis: `data/f2/raw/f2_results_fia.csv` (Scraper Output)

## F3 (bereinigte Renndaten)
- Quelle: FIA F3 Ergebnisse (ursprünglich via externem Repo gesammelt)
- Hinweis: Die Rohdaten wurden in einem separaten, älteren Repo verarbeitet.
  Aufgrund der Komplexität wurde dieses Projekt mit bereinigten Daten gestartet.
- Pfad: `data/f3/interim/f3_races_clean.csv`

## Abgeleitete Datensätze
- F1/F2/F3 Season-Features: data/*/processed/
- Master-Features (alle Serien): `data/all_series/processed/`
- Modell-Input und Splits: `data/model_input/`
