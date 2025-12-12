from pathlib import Path
import pandas as pd

OUT = Path("data/all_series/processed/all_series_master_features_core.csv")

FILES = [
    Path("data/f1/processed/f1_features_core.csv"),
    Path("data/f2/processed/f2_features.csv"),
    Path("data/f3/processed/f3_features.csv"),
]

def main() -> None:
    dfs = []
    for p in FILES:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        dfs.append(pd.read_csv(p))

    df = pd.concat(dfs, ignore_index=True)

    # --- FIX: season darf im Core-Set nicht existieren ---
    if "season" in df.columns:
        df = df.drop(columns=["season"])

    # optionale stabile Sortierung
    df = df.sort_values(["series", "year", "driver_name"], ignore_index=True)


    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved: {OUT} rows={len(df)} cols={len(df.columns)}")

if __name__ == "__main__":
    main()
