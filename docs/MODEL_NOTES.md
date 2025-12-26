# Model Notes

## Modellwahl
- Baseline: Logistische Regression (gut interpretierbar, stabil)
- Vergleich: Random Forest (nichtlinearer Benchmark)

## Trainingslogik
- Zielvariable: `f1_entry` (Eintritt in F1 nach der betrachteten Saison)
- Zeitbasierter Split: Training bis 2021, Test 2022-2023
- Datenleakage: `first_f1_year` wird konsequent aus Features entfernt

## Metriken
- ROC AUC und PR AUC (robust bei Klassenimbalance)
- Top-k Recall (praktischer Nutzen für Ranking/Scouting)

## Limitationen
- Starke Klassenimbalance (wenige positive Beispiele)
- Datenqualität und Missing Values
- Knowledge-Base-Regeln sind heuristisch
- Prototypischer Charakter, keine realen Investitionsentscheidungen
