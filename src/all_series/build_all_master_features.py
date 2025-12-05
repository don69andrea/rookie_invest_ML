from pathlib import Path
import pandas as pd
import numpy as np


F1_PATH = Path("data/f1/processed/f1_features.csv")
F2_PATH = Path("data/f2/processed/f2_features.csv")
F3_PATH = Path("data/f3/processed/f3_features.csv")

OUTPUT_DIR = Path("data/all_series/processed")
OUTPUT_PATH = OUTPUT_DIR / "all_series_master_features.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def load_f1_features(path: Path = F1_PATH) -> pd.DataFrame:
    df = _load_csv(path)

    # Harmonisierung Spaltennamen
    df["series"] = "F1"  # safety
    # F1 hat 'year' schon, Teamname heisst 'constructor_name'
    if "team_name" not in df.columns and "constructor_name" in df.columns:
        df["team_name"] = df["constructor_name"]

    return df


def load_f2_features(path: Path = F2_PATH) -> pd.DataFrame:
    df = _load_csv(path)

    # F2-Features haben bereits 'year' (aus race_date abgeleitet)
    df["series"] = "F2"  # safety
    # team_name existiert bereits
    return df


def load_f3_features(path: Path = F3_PATH) -> pd.DataFrame:
    df = _load_csv(path)

    # F3-Features benutzen 'season' als Jahres-Spalte -> in 'year' umbenennen
    if "season" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"season": "year"})

    df["series"] = "F3"  # safety

    return df


def build_all_series_master_features(
    f1_path: Path = F1_PATH,
    f2_path: Path = F2_PATH,
    f3_path: Path = F3_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """
    Führt F1-, F2- und F3-Season-Features in einer Master-Tabelle zusammen.
    - Harmonisiert Kernspalten (year, team_name)
    - Vereinheitlicht die Spaltenmenge (Union aller Features)
    - Fügt fehlende Spalten je Serie mit NaN hinzu
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f1 = load_f1_features(f1_path)
    f2 = load_f2_features(f2_path)
    f3 = load_f3_features(f3_path)

    # --- Spaltenmenge vereinheitlichen (Union) ---

    all_columns = sorted(set(f1.columns) | set(f2.columns) | set(f3.columns))

    def align_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        # Nur diese Spalten, in gleicher Reihenfolge
        return df[cols].copy()

    f1_aligned = align_columns(f1, all_columns)
    f2_aligned = align_columns(f2, all_columns)
    f3_aligned = align_columns(f3, all_columns)

    # --- Zusammenführen ---

    combined = pd.concat([f1_aligned, f2_aligned, f3_aligned], ignore_index=True)

    # Optional: sortieren nach Serie, Jahr, Fahrername
    sort_cols = [c for c in ["series", "year", "driver_name"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)

    combined.to_csv(output_path, index=False)
    print(f"✅ All-series master features written to: {output_path} "
          f"(rows={combined.shape[0]}, cols={combined.shape[1]})")

    return output_path


if __name__ == "__main__":
    build_all_series_master_features()
