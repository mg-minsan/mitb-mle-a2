"""Publish a business-facing copy of one month's gold predictions.

Keeps only customer_id and the default probability as a percentage, in one
folder across model versions.
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def main(snapshot_date, model_name, datamart=Path("datamart")):
    snapshot = datetime.strptime(snapshot_date, "%Y-%m-%d")
    stem = Path(model_name).stem
    prediction_path = (datamart / "gold/model_predictions" / stem
                       / f"{stem}_{snapshot:%Y_%m_%d}.csv")
    predictions = pd.read_csv(prediction_path)

    business = pd.DataFrame({
        "customer_id": predictions["customer_id"],
        "default_probability_pct": (predictions["model_predictions"] * 100).round(2),
    })

    directory = datamart / "gold/business_predictions"
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"business_predictions_{snapshot:%Y_%m_%d}.csv"
    business.to_csv(output_path, index=False)
    print(f"business predictions saved to: {output_path} ({len(business)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business prediction export")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date in YYYY-MM-DD format")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., credit_model_2024_01_01.pkl)")
    args = parser.parse_args()
    main(args.snapshotdate, args.model)
