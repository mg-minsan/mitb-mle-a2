"""Build a fixed PSI reference once, then monitor saved monthly predictions.

Run from the project root with PYTHONPATH=.; see PSI_MONITORING.md.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle

import pandas as pd

from utils.psi import build_reference, monitor


KEYS = ["customer_id", "snapshot_date"]


def month(value):
    parsed = datetime.strptime(value, "%Y-%m-%d").date()
    if parsed.day != 1:
        raise ValueError("Snapshot dates must be the first day of a month")
    return parsed


def read_partition(path):
    if path.is_file() and path.suffix == ".csv":
        return pd.read_csv(path)
    # Accept legacy Spark CSV directories with a .parquet suffix as well as
    # actual Spark/Pandas Parquet output.
    csv_files = sorted(path.glob("part-*.csv")) if path.is_dir() else []
    if csv_files:
        return pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
    return pd.read_parquet(path)


def features_for_month(datamart, snapshot_date):
    path = datamart / "gold/features_store" / f"gold_features_store_{snapshot_date:%Y_%m_%d}.parquet"
    frame = read_partition(path)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.date
    if frame.empty or not frame["snapshot_date"].eq(snapshot_date).all():
        raise ValueError(f"{path}: expected a nonempty partition for {snapshot_date}")
    if frame.duplicated(KEYS).any() or frame[KEYS].isna().any().any():
        raise ValueError(f"{path}: invalid or duplicate application keys")
    return frame


def reference_path(args):
    return args.model_bank / f"{Path(args.model).stem}_psi_reference.json"


def create_reference(args):
    start, end = month(args.startdate), month(args.enddate)
    if start > end:
        raise ValueError("The reference start date must not follow its end date")
    destination = reference_path(args)
    if destination.exists():
        raise FileExistsError(f"Reference already exists: {destination}; do not refit it monthly")
    # Only load trusted local model artefacts, in the model's pinned environment.
    with (args.model_bank / args.model).open("rb") as file:
        artefact = pickle.load(file)
    if end >= artefact["data_dates"]["model_train_date"].date():
        raise ValueError("Reference cohorts must precede the model's deployment date")
    dates = pd.date_range(start, end, freq="MS")
    features = pd.concat([features_for_month(args.datamart, d.date()) for d in dates], ignore_index=True)
    x = features[artefact["feature_cols"]]
    transformer = artefact["preprocessing_transformers"]["transformer"]
    scores = artefact["model"].predict_proba(transformer.transform(x))[:, 1]
    reference = build_reference(x, scores, {
        "model_name": args.model,
        "model_version": artefact["model_version"],
        "reference_type": "historical_application_cohorts",
        "deployment_date": artefact["data_dates"]["model_train_date"].date().isoformat(),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    })
    destination.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    print(f"Saved fixed reference {reference['reference_id']}: {destination} ({len(x)} rows)")


def monitor_month(args):
    snapshot_date = month(args.snapshotdate)
    reference = json.loads(reference_path(args).read_text())
    if reference["metadata"]["model_name"] != args.model:
        raise ValueError("The reference belongs to a different model")
    if snapshot_date <= month(reference["metadata"]["end_date"]):
        raise ValueError("Monitoring months must follow the reference period")
    if snapshot_date < month(reference["metadata"]["deployment_date"]):
        raise ValueError("Monitoring months must not precede model deployment")
    features = features_for_month(args.datamart, snapshot_date)
    stem = Path(args.model).stem
    prediction_path = (args.datamart / "gold/model_predictions" / stem
                       / f"{stem}_{snapshot_date:%Y_%m_%d}.csv")
    if not prediction_path.exists():
        prediction_path = prediction_path.with_suffix(".parquet")
    predictions = read_partition(prediction_path)
    predictions["snapshot_date"] = pd.to_datetime(predictions["snapshot_date"]).dt.date
    if not predictions["snapshot_date"].eq(snapshot_date).all():
        raise ValueError("Predictions contain an unexpected snapshot date")
    if not predictions["model_name"].eq(args.model).all():
        raise ValueError("Predictions contain an unexpected model version")
    # Outer one-to-one join catches missing, extra and duplicated predictions.
    aligned = features.merge(predictions[KEYS + ["model_predictions"]], on=KEYS,
                             how="outer", validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Feature rows and prediction rows do not match exactly")
    result = monitor(reference, aligned, aligned["model_predictions"])
    result.insert(0, "snapshot_date", snapshot_date)
    result.insert(1, "model_name", args.model)
    result.insert(2, "reference_start_date", month(reference["metadata"]["start_date"]))
    result.insert(3, "reference_end_date", month(reference["metadata"]["end_date"]))
    directory = args.datamart / "gold/model_monitoring" / stem
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"psi_{snapshot_date:%Y_%m_%d}.csv"
    result.to_csv(destination, index=False)
    print(result[["scope", "feature_name", "psi", "status", "current_missing_rate"]]
          .sort_values("psi", ascending=False).to_string(index=False))
    print(f"Saved {len(result)} PSI results: {destination}")


def main():
    parser = argparse.ArgumentParser(description="Fixed-reference monthly PSI monitoring")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["reference", "monitor"]:
        command = commands.add_parser(name)
        command.add_argument("--model", required=True, help="Model filename in model_bank")
        command.add_argument("--model-bank", type=Path, default=Path("model_bank"))
        command.add_argument("--datamart", type=Path, default=Path("datamart"))
        if name == "reference":
            command.add_argument("--startdate", required=True, help="First reference application month")
            command.add_argument("--enddate", required=True, help="Last reference application month")
            command.set_defaults(run=create_reference)
        else:
            command.add_argument("--snapshotdate", required=True)
            command.set_defaults(run=monitor_month)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
