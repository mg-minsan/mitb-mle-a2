"""Historical champion/challenger evaluation on identical held-out rows."""
from datetime import datetime
from pathlib import Path
import pickle

import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.metrics import brier_score_loss, roc_auc_score


def evaluate_promotion(candidate, X_oot, y_oot, model_bank):
    """Return an auditable decision; never select a future or arbitrary model."""
    cutoff = candidate['data_dates']['model_train_date']
    if cutoff < datetime(2024, 1, 1):
        raise ValueError('No supported deployment before 2024-01-01')

    def score(artefact):
        features = X_oot[artefact['feature_cols']]
        transformed = artefact['preprocessing_transformers']['transformer'].transform(features)
        probabilities = artefact['model'].predict_proba(transformed)[:, 1]
        return {
            'auc': float(roc_auc_score(y_oot, probabilities)),
            'brier': float(brier_score_loss(y_oot, probabilities)),
        }

    candidate_scores = score(candidate)
    previous_name = None
    if cutoff == datetime(2024, 1, 1):
        # Constant-risk forecast using training prevalence, not OOT prevalence.
        baseline = np.full(len(y_oot), candidate['data_stats']['y_train'])
        incumbent_scores = {
            'auc': 0.5,
            'brier': float(brier_score_loss(y_oot, baseline)),
        }
        approved = (candidate_scores['auc'] > incumbent_scores['auc']
                    and candidate_scores['brier'] <= incumbent_scores['brier'])
    else:
        previous_cutoff = cutoff - relativedelta(months=6)
        previous_name = f'credit_model_{previous_cutoff:%Y_%m_%d}.pkl'
        with (Path(model_bank) / previous_name).open('rb') as file:
            incumbent = pickle.load(file)  # Trusted local model bank only.
        if (incumbent['model_version'] != Path(previous_name).stem
                or incumbent['data_dates']['model_train_date'] != previous_cutoff):
            raise ValueError('Previous model metadata does not match historical cutoff')
        incumbent_scores = score(incumbent)
        approved = (candidate_scores['auc'] >= incumbent_scores['auc']
                    and candidate_scores['brier'] <= incumbent_scores['brier'])

    return {
        'candidate': candidate['model_version'],
        'previous_model': previous_name,
        'comparison_type': 'incumbent' if previous_name else 'first_model_baseline',
        'oot_start_date': candidate['data_dates']['oot_start_date'].isoformat(),
        'oot_end_date': candidate['data_dates']['oot_end_date'].isoformat(),
        'oot_rows': len(y_oot),
        'candidate_scores': candidate_scores,
        'incumbent_scores': incumbent_scores,
        'approved': bool(approved),
        'rule': 'AUC must not decrease; Brier must not increase (first AUC must exceed 0.5)',
    }
