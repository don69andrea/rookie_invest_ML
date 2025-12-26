import pandas as pd

# Pfade an dein Projekt anpassen
INPUT_PATH = "data/f2/interim/f2_results_fia_clean.csv"
OUTPUT_PATH = "data/f2/interim/f2_results_fia_drivers_clean.csv"

def main() -> None:
    # 1. Daten laden
    df = pd.read_csv(INPUT_PATH)

    # 2. Treibername aufraeumen
    name = df["driver_name"].astype(str).str.strip()

    # Initiale ist das erste Element
    df["driver_initial"] = (
        name.str.split()
            .str[0]
            .str.replace(".", "", regex=False)
    )

    # Nachname ist der Rest zusammengefuegt
    df["driver_last_name"] = (
        name.str.split()
            .str[1:]
            .str.join(" ")
    )

    # Optional: Reihenfolge der Spalten etwas ordnen
    cols = list(df.columns)
    # Neue Spalten direkt nach driver_name einfuegen
    for new_col in ["driver_initial", "driver_last_name"]:
        cols.remove(new_col)
        idx = cols.index("driver_name") + 1
        cols.insert(idx, new_col)

    df = df[cols]

    # 3. Speichern
    df.to_csv(OUTPUT_PATH, index=False)

    print("Fertig. Gespeichert unter:", OUTPUT_PATH)
    print(df[["driver_name", "driver_initial", "driver_last_name"]].head(10))


if __name__ == "__main__":
    main()
