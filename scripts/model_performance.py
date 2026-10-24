"""Evaluate original application predictions when their six-month labels mature.

Run from the project root with PYTHONPATH=.; --snapshotdate is the label
observation/evaluation month, not the month to score. No model is fitted or used
to rescore applicants. See MODEL_PERFORMANCE.md.
"""

import argparse
from pathlib import Path
import pickle

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.model_monitoring import KEYS, features_for_month, month, read_partition


LABEL_DEF = "30dpd_6mob"


def validate_keys(frame, population):
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"Invalid or duplicate {population} keys")


def evaluate_month(args):
    evaluation_date = month(args.snapshotdate)
    prediction_date = evaluation_date - relativedelta(months=6)
    stem = Path(args.model).stem

    # Only load trusted local artefacts in the pinned training environment.
    # Read metadata for version/target eligibility; never call predict_proba.
    with (args.model_bank / args.model).open("rb") as file:
        artefact = pickle.load(file)
    if artefact["model_version"] != stem:
        raise ValueError("The artefact version does not match the requested model")
    if artefact["label_def"] != LABEL_DEF:
        raise ValueError(f"Performance monitoring requires the {LABEL_DEF} target")
    deployment_date = pd.Timestamp(artefact["data_dates"]["model_train_date"])
    if pd.isna(deployment_date):
        raise ValueError("The model deployment date must be known")
    if prediction_date < deployment_date.date():
        raise ValueError("Predictions must not precede model deployment")

    features = features_for_month(args.datamart, prediction_date)
    prediction_path = (args.datamart / "gold/model_predictions" / stem
                       / f"{stem}_{prediction_date:%Y_%m_%d}.csv")
    if not prediction_path.exists():
        prediction_path = prediction_path.with_suffix(".parquet")
    predictions = read_partition(prediction_path)
    predictions["snapshot_date"] = pd.to_datetime(predictions["snapshot_date"]).dt.date
    if not predictions["snapshot_date"].eq(prediction_date).all():
        raise ValueError("Predictions contain an unexpected prediction date")
    if predictions["model_name"].isna().any() or not predictions["model_name"].eq(args.model).all():
        raise ValueError("Predictions contain an unexpected model version")
    validate_keys(predictions, "prediction")
    scores = pd.to_numeric(predictions["model_predictions"], errors="coerce")
    if scores.isna().any() or not (np.isfinite(scores) & scores.between(0, 1)).all():
        raise ValueError("Prediction scores must be finite probabilities in [0, 1]")
    predictions["model_predictions"] = scores
    applications = features[KEYS].merge(
        predictions[KEYS + ["model_predictions"]], on=KEYS,
        how="outer", validate="one_to_one", indicator=True,
    )
    if not applications["_merge"].eq("both").all():
        raise ValueError("Feature rows and prediction rows do not match exactly")

    label_path = (args.datamart / "gold/label_store"
                  / f"gold_label_store_{evaluation_date:%Y_%m_%d}.parquet")
    labels = read_partition(label_path)
    labels["snapshot_date"] = pd.to_datetime(labels["snapshot_date"]).dt.date
    labels["feature_snapshot_date"] = pd.to_datetime(labels["feature_snapshot_date"]).dt.date
    if not labels["snapshot_date"].eq(evaluation_date).all():
        raise ValueError("Labels contain an unexpected label observation date")
    if not labels["feature_snapshot_date"].eq(prediction_date).all():
        raise ValueError("Labels contain an unexpected label application date")
    if labels["label_def"].isna().any() or not labels["label_def"].eq(LABEL_DEF).all():
        raise ValueError("Labels contain an unexpected label definition")
    labels = labels.drop(columns="snapshot_date").rename(
        columns={"feature_snapshot_date": "snapshot_date"},
    )
    validate_keys(labels, "label")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    if labels["label"].isna().any() or not labels["label"].isin([0, 1]).all():
        raise ValueError("Labels must be nonmissing binary values (0 or 1)")

    aligned = applications[KEYS + ["model_predictions"]].merge(
        labels[KEYS + ["label"]], on=KEYS, how="outer",
        validate="one_to_one", indicator=True,
    )
    if aligned["_merge"].eq("right_only").any():
        raise ValueError("Labels contain unscored applications")
    n_predictions = len(predictions)
    n_labeled = int(aligned["_merge"].eq("both").sum())
    metrics = {
        "evaluation_date": evaluation_date,
        "prediction_snapshot_date": prediction_date,
        "model_name": args.model,
        "n_predictions": n_predictions,
        "n_labeled": n_labeled,
        "label_coverage": n_labeled / n_predictions,
        "auc": None,
        "gini": None,
        "default_rate": None,
        "mean_prediction": float(scores.mean()),
        "status": "insufficient_labels",
    }
    # Never report outcome metrics on a biased, partially labelled subset.
    if n_labeled == n_predictions:
        metrics["default_rate"] = float(aligned["label"].mean())
        metrics["status"] = "single_class"
        if aligned["label"].nunique() == 2:
            metrics["auc"] = float(roc_auc_score(aligned["label"], aligned["model_predictions"]))
            metrics["gini"] = 2 * metrics["auc"] - 1
            metrics["status"] = "ok"

    result = pd.DataFrame([metrics])
    directory = args.datamart / "gold/model_performance" / stem
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"performance_{evaluation_date:%Y_%m_%d}.csv"
    result.to_csv(destination, index=False)
    print(result.to_string(index=False))
    print(f"Saved performance results: {destination}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Delayed monthly performance for six-month default labels")
    parser.add_argument("--snapshotdate", required=True, help="Label observation/evaluation month, YYYY-MM-DD")
    parser.add_argument("--model", required=True, help="Model that generated the six-month-old predictions")
    parser.add_argument("--model-bank", type=Path, default=Path("model_bank"))
    parser.add_argument("--datamart", type=Path, default=Path("datamart"))
    evaluate_month(parser.parse_args(argv))


if __name__ == "__main__":
    main()
