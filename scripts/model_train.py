import os
import argparse
import glob
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, brier_score_loss
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV

from utils.psi import build_reference
from utils.model_promotion import evaluate_promotion
from utils.decision_threshold import select_decision_threshold


# Top 11 features by mean |SHAP| on the full 44-feature model (2024-09-01 cutoff),
# plus credit_mix, which has low SHAP but the highest gain importance.
# Cut from 44 features at ~0.002 CV AUC cost; see notebooks/model_training.ipynb.
SELECTED_FEATURES = [
    "outstanding_debt",
    "fe_5_mean_12m",
    "fe_10_mean_12m",
    "fe_9_mean_12m",
    "interest_rate",
    "delay_from_due_date",
    "fe_4_mean_12m",
    "fe_8_mean_12m",
    "fe_3_mean_12m",
    "spend_level",
    "changed_credit_limit",
    "credit_mix",
]


def build_training_config(snapshot_date):
    """Use the requested historical cutoff, never the date the task runs today."""
    train_date = datetime.strptime(snapshot_date, "%Y-%m-%d")
    if train_date.day != 1:
        raise ValueError("Training snapshot date must be the first day of a month")

    if train_date < datetime(2024, 1, 1):
        raise ValueError("First supported training cutoff is 2024-01-01")

    # Expand from the first nonempty label-observation month; never discard
    # older history when a new version is trained.
    history_start = datetime(2023, 7, 1)
    oot_period_months = 2
    oot_start_date = train_date - relativedelta(months=oot_period_months)
    train_test_period_months = (
        (oot_start_date.year - history_start.year) * 12
        + oot_start_date.month - history_start.month
    )
    return {
        "model_train_date_str": train_date.strftime("%Y-%m-%d"),
        "model_train_date": train_date,
        "train_test_period_months": train_test_period_months,
        "oot_period_months": oot_period_months,
        "oot_end_date": train_date - timedelta(days=1),
        "oot_start_date": oot_start_date,
        "train_test_end_date": oot_start_date - timedelta(days=1),
        "train_test_start_date": history_start,
        "train_test_ratio": 0.8,
    }


def validate_label_coverage(label_dates, config):
    """Fail before fitting if historical labels are missing or outside the cutoff."""
    dates = pd.to_datetime(pd.Series(label_dates))
    if dates.isna().any():
        raise ValueError("Label observation dates must not be missing")
    if ((dates < config["train_test_start_date"]) | (dates >= config["model_train_date"])).any():
        raise ValueError("Labels must be within the historical window and before the training cutoff")

    expected = set(pd.period_range(
        config["train_test_start_date"], config["oot_end_date"], freq="M",
    ).astype(str))
    observed = set(dates.dt.to_period("M").astype(str))
    missing = sorted(expected - observed)
    if missing:
        raise ValueError("Missing label months before cutoff: " + ", ".join(missing))


def main(snapshot_date):
    print('\n\n---starting job---\n\n')
    config = build_training_config(snapshot_date)

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")
    pprint.pprint(config)

    ## Load label store data

    folder_path = "datamart/gold/label_store/"
    files_list = [folder_path + os.path.basename(file) for file in glob.glob(os.path.join(folder_path, "*")) ]
    label_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    print("label_store_sdf count: ", label_store_sdf.count())
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date" ]) & (col("snapshot_date") <= config["oot_end_date" ]))
    print("excepted labels_sdf count: ", labels_sdf.count(), config["train_test_start_date"], config["oot_end_date"])


    # Load feature store data
    feature_folder_path = "datamart/gold/features_store/"
    feature_files_list = [feature_folder_path + os.path.basename(file) for file in glob.glob(os.path.join(feature_folder_path, "*")) ]
    print("feature_files_list: ", feature_files_list)
    feature_store_sdf = spark.read.option("header", "true").parquet(*feature_files_list)
    print("feature_store_sdf count: ", feature_store_sdf.count())
    # features are dated at loan start; the label store carries that date as feature_snapshot_date
    features_sdf = feature_store_sdf.withColumnRenamed("snapshot_date", "feature_snapshot_date")
    print("features_sdf count: ", features_sdf.count())

    data_pdf = labels_sdf.join(features_sdf, on=["customer_id", "feature_snapshot_date"], how="left").toPandas()
    validate_label_coverage(data_pdf["snapshot_date"], config)

    # split data into train/test and oot sets based on snapshot_date
    oot_pdf = data_pdf[
        (data_pdf["snapshot_date"] >= config["oot_start_date"].date())
        & (data_pdf["snapshot_date"] <= config["oot_end_date"].date())
    ]
    train_test_pdf = data_pdf[
        (data_pdf["snapshot_date"] >= config["train_test_start_date"].date())
        & (data_pdf["snapshot_date"] <= config["train_test_end_date"].date())
    ]
    missing_features = [c for c in SELECTED_FEATURES if c not in data_pdf.columns]
    if missing_features:
        raise ValueError("Selected features missing from the feature store: " + ", ".join(missing_features))
    feature_cols = SELECTED_FEATURES

    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    X_train_test = train_test_pdf[feature_cols]

    # no separate val set: hyperparameters are tuned with cross-validation on train
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test,
        train_test_pdf["label"],
        test_size=1 - config["train_test_ratio"],
        random_state=42,
        shuffle=True,
        stratify=train_test_pdf["label"],
    )

    print("X_train shape: ", X_train.shape[0])
    print("X_test shape: ", X_test.shape[0])
    print("X_oot shape: ", X_oot.shape[0])
    print("y_train shape: ", y_train.shape[0], round(y_train.mean(), 2))
    print("y_test shape: ", y_test.shape[0], round(y_test.mean(), 2))
    print("y_oot shape: ", y_oot.shape[0], round(y_oot.mean(), 2))

    #pre-processing:
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    print("categorical:", cat_cols)

    scaler = StandardScaler()
    transformer = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat_cols)
    ]).fit(X_train)

    X_train_processed = transformer.transform(X_train)
    X_test_processed = transformer.transform(X_test)
    X_oot_processed = transformer.transform(X_oot)

    print('X_train_processed', X_train_processed.shape[0])
    print('X_test_processed', X_test_processed.shape[0])
    print('X_oot_processed', X_oot_processed.shape[0])

    pd.DataFrame(X_train_processed, columns=transformer.get_feature_names_out())

    # train model
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

    param_dist = {
        'n_estimators': [25, 50],
        'max_depth': [2, 3],  # lower max_depth to simplify the model
        'learning_rate': [0.01, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring="roc_auc",  # uses predict_proba; make_scorer(roc_auc_score) would score hard 0/1 predictions
        n_iter=10,  # Number of iterations for random search
        cv=3,       # Number of folds in cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1   # Use all available cores
    )

    print("Starting hyperparameter search: 10 candidates, 3 folds...", flush=True)
    random_search.fit(X_train_processed, y_train)

    print("Best parameters found: ", random_search.best_params_)
    print("Best AUC score: ", random_search.best_score_)

    raw_model = random_search.best_estimator_

    # refit the tuned xgboost inside 3-fold calibration, so the calibrator only learns from out-of-fold predictions
    best_model = CalibratedClassifierCV(
        xgb.XGBClassifier(eval_metric='logloss', random_state=42, **random_search.best_params_),
        method='sigmoid',
        cv=3,
    )
    best_model.fit(X_train_processed, y_train)

    # train AUC is part in-sample, part out-of-fold for the calibrated model
    y_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
    train_auc_score = roc_auc_score(y_train, y_pred_proba)
    print("Train AUC score: ", train_auc_score)

    # This held-out split also serves as validation for threshold selection.
    # OOT remains untouched by threshold optimisation.
    y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
    test_auc_score = roc_auc_score(y_test, y_pred_proba)
    decision_threshold = select_decision_threshold(y_test, y_pred_proba)
    print("Test AUC score: ", test_auc_score)
    print("Decision threshold (validation Youden J): ", decision_threshold)

    # Evaluate the model on the oot set
    y_pred_proba = best_model.predict_proba(X_oot_processed)[:, 1]
    oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
    print("OOT AUC score: ", oot_auc_score)

    oot_brier_raw = brier_score_loss(y_oot, raw_model.predict_proba(X_oot_processed)[:, 1])
    oot_brier_score = brier_score_loss(y_oot, y_pred_proba)
    print(f"OOT Brier score: {oot_brier_score:.4f} calibrated vs {oot_brier_raw:.4f} raw")

    print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
    print("Test GINI score: ", round(2*test_auc_score-1,3))
    print("OOT GINI score: ", round(2*oot_auc_score-1,3))

    # prepare model
    model_artefact = {}

    model_artefact['model'] = best_model
    model_artefact['decision_threshold'] = decision_threshold
    model_artefact['threshold_method'] = 'youden_j'
    model_artefact['threshold_selection_split'] = 'held_out_test_as_validation'
    model_artefact['threshold_positive_class'] = 1
    model_artefact['threshold_comparison'] = '>='
    model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['transformer'] = transformer  # imputer + scaler + one-hot, fitted on train only
    model_artefact['feature_cols'] = feature_cols  # column order the transformer expects at inference
    model_artefact['label_def'] = data_pdf['label_def'].iloc[0]
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_test'] = X_test.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
    model_artefact['data_stats']['y_train'] = float(y_train.mean())
    model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
    model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
    model_artefact['results'] = {}
    model_artefact['results']['auc_train'] = train_auc_score
    model_artefact['results']['auc_test'] = test_auc_score
    model_artefact['results']['auc_oot'] = oot_auc_score
    model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
    model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
    model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)
    model_artefact['results']['brier_oot'] = oot_brier_score
    model_artefact['hp_params'] = random_search.best_params_
    model_artefact['calibration'] = {'method': 'sigmoid', 'cv': 3}
    pprint.pprint(model_artefact)

    model_bank_directory = "model_bank/"

    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)

    # Compare both models on exactly the same held-out rows before publishing.
    # Pass the full frame so the incumbent can select its own feature columns.
    promotion = evaluate_promotion(model_artefact, oot_pdf, y_oot, model_bank_directory)
    report_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '_promotion.json')
    with open(report_path, 'w') as file:
        json.dump(promotion, file, indent=2, allow_nan=False)
    pprint.pprint(promotion)
    if not promotion['approved']:
        spark.stop()
        raise ValueError(f"Model promotion rejected; candidate was not published. See {report_path}")
    model_artefact['promotion'] = promotion

    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"Model saved to {file_path}")

    # Freeze monitoring bins on raw training features before imputation.
    reference_dates = data_pdf.loc[X_train.index, "feature_snapshot_date"]
    reference = build_reference(
        X_train,
        best_model.predict_proba(X_train_processed)[:, 1],
        {
            "model_name": model_artefact['model_version'] + '.pkl',
            "model_version": model_artefact['model_version'],
            "reference_type": "training_split",
            "deployment_date": config["model_train_date_str"],
            "start_date": reference_dates.min().isoformat(),
            "end_date": reference_dates.max().isoformat(),
        },
    )
    reference_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '_psi_reference.json')
    with open(reference_path, 'w') as file:
        json.dump(reference, file, indent=2, allow_nan=False)
    print(f"PSI reference saved to {reference_path}")

    # sanity check: reload and reproduce the OOT AUC
    with open(file_path, 'rb') as file:
        loaded_model_artefact = pickle.load(file)

    X_oot_check = loaded_model_artefact['preprocessing_transformers']['transformer'].transform(X_oot[loaded_model_artefact['feature_cols']])
    y_pred_proba = loaded_model_artefact['model'].predict_proba(X_oot_check)[:, 1]
    print("Reloaded OOT AUC score: ", roc_auc_score(y_oot, y_pred_proba))

    spark.stop()
    print('\n\n---job completed---\n\n')


def cli(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the credit model")
    parser.add_argument(
        "--snapshotdate", required=True,
        help="Historical training cutoff, YYYY-MM-DD (first day of a month)",
    )
    args = parser.parse_args(argv)
    main(args.snapshotdate)


if __name__ == "__main__":
    cli()
