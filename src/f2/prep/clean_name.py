import pandas as pd

# Pfade an dein Projekt anpassen
input_path = "/Users/sheyla/Desktop/rookie_invest_ML/data/f2/processed/f2_results_fia_clean.csv"
output_path = "/Users/sheyla/Desktop/rookie_invest_ML/data/f2/processed/f2_results_fia_drivers_clean.csv"

# 1. Daten laden
df = pd.read_csv(input_path)

# 2. Treibername aufräumen
name = df["driver_name"].astype(str).str.strip()

# Initiale ist das erste Element
df["driver_initial"] = (
    name.str.split()
        .str[0]
        .str.replace(".", "", regex=False)
)

# Nachname ist der Rest zusammengefügt
df["driver_last_name"] = (
    name.str.split()
        .str[1:]
        .str.join(" ")
)

# Optional: Reihenfolge der Spalten etwas ordnen
cols = list(df.columns)
# Neue Spalten direkt nach driver_name einfügen
for new_col in ["driver_initial", "driver_last_name"]:
    cols.remove(new_col)
    idx = cols.index("driver_name") + 1
    cols.insert(idx, new_col)

df = df[cols]

# 3. Speichern
df.to_csv(output_path, index=False)

print("Fertig. Gespeichert unter:", output_path)
print(df[["driver_name", "driver_initial", "driver_last_name"]].head(10))
