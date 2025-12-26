from __future__ import annotations

from pathlib import Path
import re
import sys

import joblib
import pandas as pd


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "src").exists():
            return parent
    raise RuntimeError("Konnte 'src' Ordner nicht finden. Prüfe Projektstruktur.")


def build_validation_lookup(project_root: Path, out_path: Path) -> None:
    csv_files = list(project_root.rglob("*.csv"))

    found = []
    for fp in csv_files:
        try:
            cols = pd.read_csv(fp, nrows=1).columns.str.lower().tolist()
            if "f1_entry" in cols and "driver_code" in cols:
                found.append(fp)
        except Exception:
            pass

    if not found:
        raise FileNotFoundError(
            "Ich finde keine CSV im Projekt, die driver_code und f1_entry enthält. "
            "Dann kann ich die Validierung nicht automatisch bauen."
        )

    source = found[0]
    df = pd.read_csv(source)

    df["driver_code"] = df["driver_code"].astype(str).str.upper().str.strip()

    val = df[["driver_code", "f1_entry"]].copy()
    if "first_f1_year" in df.columns:
        val["first_f1_year"] = df["first_f1_year"]

    val = val.dropna(subset=["driver_code"]).drop_duplicates(subset=["driver_code"], keep="last")

    out_path.write_text(val.to_csv(index=False), encoding="utf-8")
    print("validation_lookup.csv geschrieben nach:", out_path.resolve())
    print("Rows:", len(val), "Cols:", val.columns.tolist())


def main() -> None:
    here = Path.cwd()
    project_root = find_project_root(here)

    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    print("Projekt-Root:", project_root)
    print("src im sys.path:", str(src_path) in sys.path)

    demo_root = project_root / "demo"
    input_dir = demo_root / "input"
    output_dir = demo_root / "output"
    artifact_dir = demo_root / "artifacts"

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob("*.csv"))
    if len(input_files) != 1:
        names = ", ".join(sorted(p.name for p in input_files)) or "keine"
        raise ValueError(
            "Bitte GENAU eine CSV in demo/input ablegen. "
            f"Gefunden: {len(input_files)} ({names})"
        )

    input_path = input_files[0]
    model_path = artifact_dir / "logreg_model.joblib"
    drop_cols_path = artifact_dir / "drop_cols.txt"
    validation_lookup_path = artifact_dir / "validation_lookup.csv"

    print("Input:", input_path.name)
    print("Output Ordner:", output_dir.resolve())

    m = re.search(r"(19|20)\\d{2}", input_path.name)
    year_label = m.group(0) if m else "Unknown Year"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    build_validation_lookup(project_root, validation_lookup_path)

    df_in = pd.read_csv(input_path)
    print("Geladene Fahrer:", len(df_in))

    logreg_model = joblib.load(model_path)

    drop_cols = set(drop_cols_path.read_text(encoding="utf-8").splitlines())
    drop_cols = {c.strip() for c in drop_cols if c.strip()}
    print("Drop columns:", drop_cols)

    X = df_in.drop(columns=list(drop_cols), errors="ignore")
    proba = logreg_model.predict_proba(X)[:, 1]

    df_rank = df_in.copy()
    df_rank["predicted_probability"] = proba

    df_rank = (
        df_rank.sort_values("predicted_probability", ascending=False)
        .reset_index(drop=True)
    )

    # Top-N + ex-post Hit + HTML
    top_n = 20
    out_path = output_dir / "top_candidates.html"
    title = f"Rookie Invest Prototype Demo – Top Kandidaten {year_label}"

    tbl = df_rank.head(top_n).copy().reset_index(drop=True)

    if "predicted_probability" in tbl.columns:
        tbl["predicted_probability"] = tbl["predicted_probability"].map(
            lambda x: f"{float(x) * 100:.1f}%"
        )

    is_hit = pd.Series([False] * len(tbl), index=tbl.index)

    if validation_lookup_path.exists() and "driver_code" in tbl.columns:
        val = pd.read_csv(validation_lookup_path)
        if {"driver_code", "f1_entry"}.issubset(val.columns):
            val_codes = val["driver_code"].astype(str).str.upper().str.strip()
            val_entries = (
                val["f1_entry"]
                .fillna(False)
                .astype(str).str.strip().str.lower()
                .isin(["true", "1", "yes", "y", "t"])
            )
            lookup = pd.Series(val_entries.values, index=val_codes.values)
            tbl_codes = tbl["driver_code"].astype(str).str.upper().str.strip()
            is_hit = tbl_codes.map(lookup).fillna(False).astype(bool)

    preferred = [
        "driver_name",
        "driver_code",
        "series",
        "year",
        "team_name",
        "predicted_probability",
    ]
    tbl_out = tbl[[c for c in preferred if c in tbl.columns]].copy()

    hit_bg = "#d9f0e0"

    def highlight_hits(row: pd.Series) -> list[str]:
        if is_hit.loc[row.name]:
            return [f"background-color: {hit_bg}"] * len(row)
        return [""] * len(row)

    styled = (
        tbl_out.style
        .apply(highlight_hits, axis=1)
        .hide(axis="index")
        .set_table_styles(
            [
                {"selector": "th", "props": [("background-color", "#111"), ("color", "white"), ("padding", "10px")]},
                {"selector": "td", "props": [("padding", "10px"), ("border-bottom", "1px solid #eee")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
            ]
        )
    )

    html = f"""
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
</head>
<body style="font-family: Arial, sans-serif; margin: 24px;">
  <h2>{title}</h2>
  <p>Ranking basiert auf Modellwahrscheinlichkeit. Input enthält keine Information über F1-Eintritt.</p>
  <p><small>Grün markiert: bestätigter F1-Einstieg ex post, nicht Teil des Modell Inputs.</small></p>
  {styled.to_html()}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print("HTML erzeugt:", out_path.resolve())

    # Hybrid Output (Anzeige): ML Prediction + Knowledge Base Context
    from knowledge_base.racing_intelligence_engine import RacingIntelligenceEngine

    base = tbl.copy()
    if "predicted_probability" not in base.columns:
        if "ml_probability" in base.columns:
            base["predicted_probability"] = base["ml_probability"]
        else:
            raise ValueError(
                "Keine Spalte 'predicted_probability' gefunden. Prüfe, wie du die Prediction-Spalte nennst."
            )

    kb_engine = RacingIntelligenceEngine()

    def _safe_get(row: pd.Series, key: str, default):
        v = row.get(key, default)
        if pd.isna(v):
            return default
        return v

    def kb_context(row: pd.Series) -> dict:
        driver_input = {
            "age": int(_safe_get(row, "age", 20)),
            "nationality": str(_safe_get(row, "nationality", "unknown")),
            "superlicense_points": float(_safe_get(row, "superlicense_points", 0)),
            "junior_series_years": float(_safe_get(row, "junior_series_years", 0)),
            "years_in_f3": float(_safe_get(row, "years_in_f3", 0)),
            "previous_series": str(_safe_get(row, "previous_series", "")),
            "social_media_behavior": str(_safe_get(row, "social_media_behavior", "neutral")),
            "weight_kg": float(_safe_get(row, "weight_kg", 70)),
            "neck_cm": float(_safe_get(row, "neck_cm", 42)),
            "sponsor_capital_chf": float(_safe_get(row, "sponsor_capital_chf", 0)),
        }

        team_input = {"team_name": str(_safe_get(row, "team_name", ""))}
        vehicle_input = {"engine_status": "ok", "drs_active": False, "tire_status": "ok"}

        return kb_engine.generate_full_profile(driver_input, team_input, vehicle_input)

    kb_df = base.apply(kb_context, axis=1, result_type="expand")
    hybrid = pd.concat([base.reset_index(drop=True), kb_df.reset_index(drop=True)], axis=1)

    preferred_cols = [
        "driver_name",
        "driver_code",
        "series",
        "year",
        "team_name",
        "predicted_probability",
        "financial_viability",
        "team_political_power",
        "f1_marketing_boost",
        "phys_neck_strength",
        "f3_pathway_score",
        "f1_qualified",
    ]

    show_cols = [c for c in preferred_cols if c in hybrid.columns]
    hybrid = hybrid.sort_values("predicted_probability", ascending=False)
    hybrid_view = hybrid[show_cols].head(25).copy()

    pp = hybrid_view["predicted_probability"].astype(str).str.replace("%", "", regex=False)
    pp = pd.to_numeric(pp, errors="coerce")

    hybrid_view = hybrid_view.drop(columns=["predicted_probability"])

    out_path2 = output_dir / "top_candidates_with_context.html"
    styled = (
        hybrid_view.style
        .format({"predicted_probability_pct": "{:.1f}%"})
        .set_table_styles(
            [
                {"selector": "th", "props": [("background-color", "#111827"), ("color", "white"), ("padding", "8px"), ("text-align", "left")]},
                {"selector": "td", "props": [("padding", "8px"), ("border", "1px solid #e5e7eb")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("font-family", "Arial"), ("font-size", "12px")]},
            ]
        )
    )

    out_path2.write_text(styled.to_html(), encoding="utf-8")
    print("Hybrid HTML erzeugt:", out_path2.resolve())


if __name__ == "__main__":
    main()
