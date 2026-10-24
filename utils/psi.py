"""Fixed-reference PSI for raw features and model default probabilities.

No labels are needed. Bin definitions are fitted on the reference only.
"""

import hashlib
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _numeric_values(series):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"{series.name}: infinite values are invalid inputs")
    return values


def _counts(series, specification):
    if specification["kind"] == "numeric":
        values = _numeric_values(series)
        cuts = specification["cut_points"]
        assignments = np.searchsorted(cuts, values, side="right")
        # The final bucket is reserved for missing values.
        assignments[np.isnan(values)] = len(cuts) + 1
        return np.bincount(assignments, minlength=len(cuts) + 2)

    categories = specification["categories"]
    values = series.astype("string")
    mapping = {value: index for index, value in enumerate(categories)}
    assignments = values.map(mapping).fillna(len(categories)).to_numpy(dtype=int)
    # Unknown and missing categories have distinct, permanent buckets.
    assignments[values.isna().to_numpy()] = len(categories) + 1
    return np.bincount(assignments, minlength=len(categories) + 2)


def _specification(series, n_bins):
    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        values = _numeric_values(series)
        observed = values[~np.isnan(values)]
        unique = np.unique(observed)
        cuts = (np.unique(np.quantile(observed, np.linspace(0, 1, n_bins + 1)[1:-1]))
                if len(observed) else np.array([]))
        if 1 < len(unique) <= n_bins:
            # Discrete features (especially imbalanced 0/1 flags) retain one
            # bucket per observed value rather than collapsing tied quantiles.
            cuts = unique[1:]
        elif len(cuts):
            # A large mass at the minimum must not share a single bucket with
            # every larger value when quantiles coincide with that minimum.
            cuts = np.unique(np.where(cuts == unique[0], np.nextafter(unique[0], np.inf), cuts))
        # Isolate a constant reference in its own bucket, so changes on either
        # side (including 1 -> 0 flags) are visible.
        if len(unique) == 1:
            cuts = np.array([unique[0], np.nextafter(unique[0], np.inf)])
        return {"kind": "numeric", "cut_points": cuts.tolist()}
    return {
        "kind": "categorical",
        "categories": sorted(series.dropna().astype(str).unique().tolist()),
    }


def _validate_scores(scores, expected_rows):
    scores = pd.Series(scores, name="model_predictions").reset_index(drop=True)
    values = _numeric_values(scores)
    if len(scores) != expected_rows or not np.isfinite(values).all():
        raise ValueError("Expected one finite prediction per feature row")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Default probabilities must be between 0 and 1")
    return scores


def build_reference(features, scores, metadata, n_bins=10):
    """Build a JSON-serialisable baseline from pre-imputation features.

    metadata should identify the model and the historical reference population.
    Store the returned object once; never refit it on each inference month.
    """
    if features.empty:
        raise ValueError("The reference dataset must not be empty")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    scores = _validate_scores(scores, len(features))
    reference = {
        "schema_version": 1,
        "metadata": metadata,
        "row_count": len(features),
        "features": {},
    }
    for name in features.columns:
        spec = _specification(features[name], n_bins)
        spec["counts"] = _counts(features[name], spec).tolist()
        reference["features"][name] = spec
    score_spec = {"kind": "numeric", "cut_points": np.linspace(0.1, 0.9, 9).tolist()}
    score_spec["counts"] = _counts(scores, score_spec).tolist()
    reference["prediction"] = score_spec
    encoded = json.dumps(reference, sort_keys=True, allow_nan=False).encode()
    reference["reference_id"] = hashlib.sha256(encoded).hexdigest()[:16]
    return reference


def population_stability_index(reference_counts, current_counts, epsilon=1e-6):
    """PSI from counts in identical buckets, including zero-count buckets.

    Floor proportions at epsilon, then renormalise to keep valid distributions.
    """
    p = np.asarray(reference_counts, dtype=float)
    q = np.asarray(current_counts, dtype=float)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("Reference and current bucket counts must have the same shape")
    if not 0 < epsilon < 1 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Counts must be finite and epsilon must be between 0 and 1")
    if (p < 0).any() or (q < 0).any() or p.sum() == 0 or q.sum() == 0:
        raise ValueError("Counts must be nonnegative with nonempty populations")
    p = np.maximum(p / p.sum(), epsilon)
    q = np.maximum(q / q.sum(), epsilon)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def monitor(reference, features, scores):
    """Return one summary row per input feature plus one prediction-score row."""
    if features.empty:
        raise ValueError("The current dataset must not be empty; check source coverage")
    missing = set(reference["features"]) - set(features.columns)
    if missing:
        raise ValueError(f"Required model features are missing: {sorted(missing)}")
    scores = _validate_scores(scores, len(features))
    rows = []
    specifications = [("feature", name, spec, features[name])
                      for name, spec in reference["features"].items()]
    specifications.append(("prediction", "model_predictions", reference["prediction"], scores))
    for scope, name, spec, series in specifications:
        counts = _counts(series, spec)
        baseline = np.asarray(spec["counts"])
        psi = population_stability_index(baseline, counts)
        rows.append({
            "reference_id": reference["reference_id"],
            "scope": scope,
            "feature_name": name,
            "feature_type": spec["kind"],
            "reference_row_count": int(baseline.sum()),
            "current_row_count": len(features),
            "psi": psi,
            "status": "alert" if psi >= 0.25 else "warning" if psi >= 0.10 else "stable",
            "reference_missing_rate": float(baseline[-1] / baseline.sum()),
            "current_missing_rate": float(counts[-1] / counts.sum()),
            "reference_unknown_rate": (float(baseline[-2] / baseline.sum())
                                       if spec["kind"] == "categorical" else 0.0),
            "current_unknown_rate": (float(counts[-2] / counts.sum())
                                     if spec["kind"] == "categorical" else 0.0),
        })
    return pd.DataFrame(rows)
