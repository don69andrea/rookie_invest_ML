# Rookie Invest – Feature Engineering Pipeline (F1 / F2 / F3)

Dieses Repository enthält die **vollständige Daten- und Feature-Pipeline** für das Projekt *Rookie Invest*.  
Ziel ist es, **vergleichbare Fahrer-Season-Features** über **Formula 1, Formula 2 und Formula 3** zu erzeugen und diese in einem gemeinsamen Master-Datensatz zusammenzuführen.

Der Fokus liegt auf:
- reproduzierbarer Feature-Erstellung
- klarer Trennung von Vorbereitung und produktiver Feature-Logik
- konsistenten Feature-Namen über alle Serien hinweg

---

## Projektüberblick

**Input:**
- Rohdaten aus verschiedenen Quellen (FIA, Kaggle, Scraping)
- Serien-spezifische Renn- und Ergebnisdaten

**Output:**
- Serien-spezifische Feature-Datensätze (`f1_features.csv`, `f2_features.csv`, `f3_features.csv`)
- Ein **gemeinsamer Core-Datensatz** für alle Serien:
  - `all_series_master_features_core.csv`

---

## Ordnerstruktur

```text
src/
├── all_series/
│   ├── build_all_master_features.py
│   └── build_all_master_features_core.py
│
├── analysis/
│   └── check_f3_season_features.py
│
├── common/
│   ├── features.py
│   ├── io.py
│   └── preprocessing.py
│
├── f1/
│   ├── build/
│   │   └── build_features.py
│   └── prep/
│       ├── ingest.py
│       └── clean.py
│
├── f2/
│   ├── build/
│   │   └── build_features.py
│   └── prep/
│       ├── ingest_fia.py
│       ├── clean_f2_results.py
│       └── clean_name.py
│
├── f3/
│   ├── build/
│   │   ├── build_features.py
│   │   └── build_features_advanced.py
│
└── schema/
    └── core_features.py
