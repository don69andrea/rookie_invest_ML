from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


def export_demo_artifacts(
    demo_df: pd.DataFrame,
    drop_cols: set[str] | list[str],
    model,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    project_root: Path | None = None,
    top_n: int = 20,
) -> None:
    if project_root is None:
        project_root = Path.cwd().parent

    export_dir = project_root / "exports"
    export_dir.mkdir(exist_ok=True)

    demo_dir = project_root / "demo"
    demo_input_by_year = demo_dir / "input_by_year"
    demo_input_by_year.mkdir(parents=True, exist_ok=True)

    artifacts_dir = demo_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # HTML ranking export
    out_path = export_dir / "top_candidates_2022_2023.html"
    tbl = demo_df.head(top_n).copy()
    if "predicted_probability" in tbl.columns:
        tbl["predicted_probability"] = tbl["predicted_probability"].map(
            lambda x: f"{x:.3f}"
        )
    if "actual_label" in tbl.columns:
        tbl["hit"] = tbl["actual_label"].map(lambda x: "✅" if int(x) == 1 else "")

    cols = [
        c
        for c in [
            "driver_name",
            "driver_code",
            "series",
            "year",
            "team_name",
            "predicted_probability",
            "hit",
        ]
        if c in tbl.columns
    ]
    tbl = tbl[cols]

    html = f"""
<html>
<head>
  <meta charset="utf-8"/>
  <title>Rookie Invest Demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h2 {{ margin-bottom: 12px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f5f5f5; }}
    tr:nth-child(even) {{ background-color: #fafafa; }}
  </style>
</head>
<body>
  <h2>Top {top_n} Candidates (2022-2023)</h2>
  {tbl.to_html(index=False, escape=False)}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")

    # Demo CSV exports (nicht in demo/input, damit der Runner sauber bleibt)
    demo_input_by_year.joinpath("test_drivers_2022_2023.csv").write_text(
        test_df.to_csv(index=False),
        encoding="utf-8",
    )

    demo_cols = [
        "series",
        "year",
        "driver_name",
        "driver_code",
        "team_name",
        "n_races",
        "total_points",
        "avg_points",
        "avg_finish",
        "wins",
        "podiums",
    ]
    demo_cols = [c for c in demo_cols if c in test_df.columns]
    demo_input_by_year.joinpath("test_drivers_2022_2023_minimal.csv").write_text(
        test_df[demo_cols].to_csv(index=False),
        encoding="utf-8",
    )

    # Input by year (labels removed)
    forbidden = {"f1_entry", "first_f1_year"}
    for year in sorted(full_df["year"].unique()):
        season_df = full_df[full_df["year"] == year].copy()
        season_df = season_df.drop(
            columns=[c for c in forbidden if c in season_df.columns], errors="ignore"
        )
        out_file = demo_input_by_year / f"drivers_{year}.csv"
        out_file.write_text(season_df.to_csv(index=False), encoding="utf-8")

    # Model + metadata
    model_path = artifacts_dir / "logreg_model.joblib"
    joblib.dump(model, model_path)

    drop_cols_path = artifacts_dir / "drop_cols.txt"
    drop_cols_path.write_text("\n".join(sorted(list(drop_cols))), encoding="utf-8")
