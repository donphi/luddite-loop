#!/usr/bin/env python3
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
