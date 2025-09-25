#!/usr/bin/env python3
# ============================================================================
# FILE: field_check.py
# LOCATION: pipelines/01-data_ingestion/02-restructure_ukb_showcase_data/scripts/field_check.py
# PIPELINE POSITION: Main Pipeline 01 → Sub-Pipeline 02
# PURPOSE: Generates category summary statistics from processed UK Biobank field data
# ============================================================================

"""
MODULE OVERVIEW:
This utility script creates a summary of fields per category from the processed ukb_fields.csv file.
It groups fields by category_title and calculates aggregate statistics including:
- Number of fields per category
- Average and sum of participant counts
- Provides insights into data distribution across categories

DEPENDENCIES:
- pandas==2.1.4
- pathlib (standard library)
"""

import pandas as pd
from pathlib import Path

# Try common locations for the output folder
candidates = [Path("./"), Path("../output")]
outdir = next((p for p in candidates if (p / "ukb_fields.csv").exists()), None)

if outdir is None:
    raise FileNotFoundError(
        "Couldn't find 'ukb_fields.csv' in ./output or ../output.\n"
        "Put the file in one of those, or adjust the paths below."
    )

inp = outdir / "ukb_fields.csv"
out = outdir / "category_summary.csv"

df = pd.read_csv(inp)

summary = (
    df.groupby("category_title", dropna=False)
      .agg(
          fields_per_category=("field_id", "count"),
          avg_num_participants=("num_participants", "mean"),
          sum_num_participants=("num_participants", "sum"),
      )
      .reset_index()
)

summary["avg_num_participants"] = summary["avg_num_participants"].round(1)

summary.to_csv(out, index=False)
print(f"✅ Wrote {out}")
