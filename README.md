# rookie_invest_ML

## Projektübersicht
Dieses Repo baut eine datengetriebene Pipeline rund um Nachwuchsfahrer in den
Serien F1, F2 und F3. Es kombiniert klassische Renn-Datenaggregation mit einer
Knowledge Base, die Reglements, Fahrerprofile und Teamkontexte in Features für
ein ML-Modell übersetzt.

Ziel: aus Rohdaten und Regeln reproduzierbare Feature-Sets zu erstellen, die
später für Ranking, Prognosen oder Scouting genutzt werden können.

## Projektbeschreibung (Schulprojekt)
Die Verbindung von Sport und Datenanalyse gewinnt an Bedeutung, da Entscheidungen
über Leistung, Karriere und wirtschaftliches Potenzial zunehmend datenbasiert
getroffen werden. Gleichzeitig entstehen neue Investitionsmodelle, bei denen in
junge Talente investiert wird, um später am Erfolg zu partizipieren. Machine
Learning in Kombination mit wissensbasierten Systemen ermöglicht es, das
Potenzial von Athleten fundierter einzuschätzen und Sponsoring-Entscheidungen
transparenter zu unterstützen.

### Ausgangslage
Der Einstieg in den Profisport stellt für junge Talente eine grosse Chance dar,
ist jedoch gleichzeitig mit erheblichen Risiken verbunden. Für Sponsoren,
Investoren und Sportorganisationen stellt sich die Frage, welche Talente ein
langfristiges Engagement rechtfertigen.

### Problemstellung
Die Beurteilung des Potenzials von Nachwuchstalenten erfolgt heute häufig
subjektiv und basiert auf individuellen Einschätzungen. Obwohl umfangreiche
Leistungsdaten verfügbar sind, werden sie nur begrenzt systematisch genutzt.
Wirtschaftliche und kontextbezogene Faktoren wie Vermarktungspotenzial oder
Verletzungsrisiken bleiben oft unzureichend berücksichtigt.

### Zentrale Fragestellung
Ziel ist die Entwicklung eines hybriden Entscheidungsmodells, das Machine
Learning mit wissensbasierten Regeln kombiniert und Sponsoren sowie Investoren
bei der Auswahl von Nachwuchstalenten unterstützt.

Forschungsfrage:
Wie können Machine Learning und wissensbasierte Systeme kombiniert werden, um
das Sponsoringpotenzial von Nachwuchstalenten in ausgewählten Sportarten
zuverlässig vorherzusagen?

Unterfragen:
- Datenbasis: Welche sportlichen, wirtschaftlichen und kontextbezogenen Faktoren
  eignen sich als Prädiktoren für den langfristigen Erfolg von Nachwuchstalenten
  in der Formel 1?
- Machine Learning: Welche ML-Modelle liefern die besten Vorhersageergebnisse
  für die Erfolgswahrscheinlichkeit von Athleten?
- Wissensbasierte Systeme: Welche Expertenregeln können zusätzlich zum
  ML-Modell eingesetzt werden, um die Prognose zu verbessern?
- Hybridmodelle: Wie lassen sich datengetriebene Vorhersagen und
  wissensbasierte Regeln zu einem hybriden Entscheidungsmodell kombinieren?

### Abgrenzung
Dieses Projekt fokussiert sich ausschliesslich auf die Analyse von
Nachwuchstalenten im Kontext der Formel 1. Es werden keine realen Investitions-
oder Sponsoringentscheidungen getroffen, sondern ausschliesslich ein
prototypisches Entscheidungsmodell entwickelt. Zudem erhebt das Modell keinen
Anspruch auf personenbezogene Bewertungen oder Prognosen einzelner realer
Athleten, sondern dient ausschliesslich als konzeptionelle und methodische
Entscheidungsunterstützung.

## Setup
Die Abhängigkeiten sind in `pyproject.toml` gepflegt.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Reproduzierbarkeit:
- `docs/REPRODUCIBILITY.md`

## Inhalte auf einen Blick
- F1: Rohdaten-Join -> Cleaning -> Season-Features (inkl. Core-Featureset).
- F2: FIA-Ergebnisse scrapen -> Cleaning -> Season-Features.
- F3: Bereinigte Race-Daten -> Season-Features (Basic + Advanced).
- Serien-Merge: F1/F2/F3 in einer Master-Tabelle.
- Knowledge Base: regelbasierte Feature-Generierung für Fahrer/Team/Fahrzeug.
- Demo-Runner für HTML-Ausgaben (ML-only + Hybrid).

## Datenpipeline (Kurzfassung)
### F1 Pipeline (Kaggle F1 Schema)
```bash
python src/f1/prep/ingest.py
python src/f1/prep/clean.py
python src/f1/build/build_features.py
```

### F2 Pipeline (FIA Ergebnisse)
Voraussetzung: `data/f2/raw/f2_dataset_manuell.xlsx`
```bash
python src/f2/prep/ingest_fia.py
python src/f2/prep/clean_f2_results.py
python src/f2/prep/clean_name.py
python src/f2/build/build_features.py
```

### F3 Pipeline
Voraussetzung: `data/f3/interim/f3_races_clean.csv`
```bash
python src/f3/build/build_race_features.py
python src/f3/build/build_features.py
python src/f3/analysis/build_features_advanced.py
```

### Serien-Merge
```bash
python src/all_series/build_all_master_features.py
python src/all_series/build_all_master_features_core.py
```

## Knowledge Base
Die Regeln liegen in `src/knowledge_base/racing_criteria.json`. Die Engine
erzeugt Feature-Flags für Qualifikation, Biometrie, Team-Fit und Telemetrie.
```bash
python src/demo/run_kb_demo.py
```

## Demo (Präsentation)
Die Demo erwartet genau eine CSV in `demo/input/`. Daraus wird das Ranking
generiert und als HTML gespeichert.
```bash
python src/demo/run_demo.py
```

Outputs:
- `demo/output/top_candidates.html`
- `demo/output/top_candidates_with_context.html`

Hinweis: Exporte aus dem Training (Notebook) schreiben nach
`demo/input_by_year/` und beeinflussen den Demo-Runner nicht.

## Notebooks
Die Referenz-Notebooks für die Dokumentation:
- `notebooks/01_data_collection_and_features.ipynb`
- `notebooks/02_labeling_and_split.ipynb`
- `notebooks/03_baseline_and_demo.ipynb`
- `notebooks/04_model_comparison.ipynb`

Übersicht:
- `docs/NOTEBOOKS.md`

## Projektstruktur (kurz)
- `src/f1`, `src/f2`, `src/f3`: Datenaufbereitung und Feature-Builds
- `src/all_series`: Zusammenführen der Serien
- `src/knowledge_base`: Regeln + Engine
- `src/schema`: Core-Feature-Schema
- `data/`: Rohdaten, Zwischenstände und Outputs
- `demo/`: Inputs, Outputs, Artefakte für die Präsentation
- `docs/`: Dokumentation und Reproducibility

## Hinweise
- Einige Configs in `configs/` sind aktuell leer und dienen als Platzhalter.
- Die Skripte erwarten bestimmte Spaltennamen; siehe Module unter
  `src/f1/prep`, `src/f2/prep`, `src/f3/build`.
