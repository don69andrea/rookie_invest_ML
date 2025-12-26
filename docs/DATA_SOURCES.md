# Data Sources

Diese Datei fasst die verwendeten Datenqüllen zusammen und beschreibt, wo sie
im Projekt abgelegt werden. So bleibt nachvollziehbar, welche Rohdaten in die
Pipeline einfliessen und welche Outputs daraus entstehen.

## F1 (Kaggle Schema)
- Qülle: F1 Kaggle Datensatz (klassisches Schema)
- Pfad: `data/f1/raw/`
- Erwartete Dateien: races.csv, results.csv, drivers.csv,
  constructors.csv, circuits.csv, status.csv (optional weitere)

## F2 (FIA Ergebnisse)
- Qülle: FIA Formula 2 Results (Scraper)
- Pfad Meta: `data/f2/raw/f2_dataset_manuell.xlsx`
- Ergebnis: `data/f2/raw/f2_results_fia.csv` (Scraper Output)

## F3 (bereinigte Renndaten)
- Qülle: bereinigte F3 Race-Daten (bereits vorbereitet)
- Pfad: `data/f3/interim/f3_races_clean.csv`

## Abgeleitete Datensätze
- F1/F2/F3 Season-Features: data/*/processed/
- Master-Features (alle Serien): `data/all_series/processed/`
- Modell-Input und Splits: `data/model_input/`
