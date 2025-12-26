# rookie_invest_ML

## Projektuebersicht
Dieses Repo baut eine datengetriebene Pipeline rund um Nachwuchsfahrer in den
Serien F1, F2 und F3. Es kombiniert klassische Renn-Datenaggregation mit einer
Knowledge Base, die Reglements, Fahrerprofile und Teamkontexte in Features fuer
ein ML-Modell uebersetzt.

Ziel: aus Rohdaten und Regeln reproduzierbare Feature-Sets zu erstellen, die
spaeter fuer Ranking, Prognosen oder Scouting genutzt werden koennen.

## Projektbeschreibung (Schulprojekt)
Die Verbindung von Sport und Datenanalyse gewinnt an Bedeutung, da Entscheidungen
ueber Leistung, Karriere und wirtschaftliches Potenzial zunehmend datenbasiert
getroffen werden. Gleichzeitig entstehen neue Investitionsmodelle, bei denen in
junge Talente investiert wird, um spaeter am Erfolg zu partizipieren. Machine
Learning in Kombination mit wissensbasierten Systemen ermoeglicht es, das
Potenzial von Athleten fundierter einzuschaetzen und Sponsoring-Entscheidungen
transparenter zu unterstuetzen.

### Ausgangslage
Der Einstieg in den Profisport stellt fuer junge Talente eine grosse Chance dar,
ist jedoch gleichzeitig mit erheblichen Risiken verbunden. Fuer Sponsoren,
Investoren und Sportorganisationen stellt sich die Frage, welche Talente ein
langfristiges Engagement rechtfertigen.

### Problemstellung
Die Beurteilung des Potenzials von Nachwuchstalenten erfolgt heute haeufig
subjektiv und basiert auf individuellen Einschaetzungen. Obwohl umfangreiche
Leistungsdaten verfuegbar sind, werden sie nur begrenzt systematisch genutzt.
Wirtschaftliche und kontextbezogene Faktoren wie Vermarktungspotenzial oder
Verletzungsrisiken bleiben oft unzureichend beruecksichtigt.

### Zentrale Fragestellung
Ziel ist die Entwicklung eines hybriden Entscheidungsmodells, das Machine
Learning mit wissensbasierten Regeln kombiniert und Sponsoren sowie Investoren
bei der Auswahl von Nachwuchstalenten unterstuetzt.

Forschungsfrage:
Wie koennen Machine Learning und wissensbasierte Systeme kombiniert werden, um
das Sponsoringpotenzial von Nachwuchstalenten in ausgewaehlten Sportarten
zuverlaessig vorherzusagen?

Unterfragen:
- Datenbasis: Welche sportlichen, wirtschaftlichen und kontextbezogenen Faktoren
  eignen sich als Praediktoren fuer den langfristigen Erfolg von Nachwuchstalenten
  in der Formel 1?
- Machine Learning: Welche ML-Modelle liefern die besten Vorhersageergebnisse
  fuer die Erfolgswahrscheinlichkeit von Athleten?
- Wissensbasierte Systeme: Welche Expertenregeln koennen zusaetzlich zum
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
Entscheidungsunterstuetzung.

## Setup
Die Abhaengigkeiten sind in `pyproject.toml` gepflegt.
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
- Knowledge Base: regelbasierte Feature-Generierung fuer Fahrer/Team/Fahrzeug.
- Demo-Runner fuer HTML-Ausgaben (ML-only + Hybrid).

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
erzeugt Feature-Flags fuer Qualifikation, Biometrie, Team-Fit und Telemetrie.
```bash
python src/demo/run_kb_demo.py
```

## Demo (Praesentation)
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
Die Referenz-Notebooks fuer die Dokumentation:
- `notebooks/01_data_collection_and_features.ipynb`
- `notebooks/02_labeling_and_split.ipynb`
- `notebooks/03_baseline_and_demo.ipynb`
- `notebooks/04_model_comparison.ipynb`

Uebersicht:
- `docs/NOTEBOOKS.md`

## Projektstruktur (kurz)
- `src/f1`, `src/f2`, `src/f3`: Datenaufbereitung und Feature-Builds
- `src/all_series`: Zusammenfuehren der Serien
- `src/knowledge_base`: Regeln + Engine
- `src/schema`: Core-Feature-Schema
- `data/`: Rohdaten, Zwischenstaende und Outputs
- `demo/`: Inputs, Outputs, Artefakte fuer die Praesentation
- `docs/`: Dokumentation und Reproducibility

## Hinweise
- Einige Configs in `configs/` sind aktuell leer und dienen als Platzhalter.
- Die Skripte erwarten bestimmte Spaltennamen; siehe Module unter
  `src/f1/prep`, `src/f2/prep`, `src/f3/build`.
