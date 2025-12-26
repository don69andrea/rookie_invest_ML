# Reproducibility

Ziel: einen klaren Ablauf, um Datenpipelines, Modellinputs und Demo-Outputs
reproduzieren zu können.

## 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2) Datenpipeline (F1/F2/F3)
### F1
```bash
python src/f1/prep/ingest.py
python src/f1/prep/clean.py
python src/f1/build/build_features.py
```

### F2 (FIA Scraper)
Voraussetzung: `data/f2/raw/f2_dataset_manuell.xlsx`
```bash
python src/f2/prep/ingest_fia.py
python src/f2/prep/clean_f2_results.py
python src/f2/prep/clean_name.py
python src/f2/build/build_features.py
```

### F3
Voraussetzung: `data/f3/interim/f3_races_clean.csv`
```bash
python src/f3/build/build_race_features.py
python src/f3/build/build_features.py
python src/f3/analysis/build_features_advanced.py
```

### All Series
```bash
python src/all_series/build_all_master_features.py
python src/all_series/build_all_master_features_core.py
```

## 3) Labeling und Splits
Notebook:
- `notebooks/02_labeling_and_split.ipynb`

Outputs:
- `data/model_input/f2_f3_features_with_f1_label.csv`
- `data/model_input/splits/train_upto_2021.csv`
- `data/model_input/splits/test_after_2021.csv`

## 4) Baseline + Demo Exports
Notebook:
- `notebooks/03_baseline_and_demo.ipynb`

Outputs (Training/Exports):
- `demo/artifacts/logreg_model.joblib`
- `demo/artifacts/drop_cols.txt`
- `demo/input_by_year/` (CSV Dateien)

## 5) Präsentation Demo
Voraussetzung: genau eine CSV in `demo/input/` (manüll).
```bash
python src/demo/run_demo.py
```

Outputs:
- `demo/output/top_candidates.html`
- `demo/output/top_candidates_with_context.html`
