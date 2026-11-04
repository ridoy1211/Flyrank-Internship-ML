"""Stage 1: compare all candidates; select only within the outer training clients.

The original February/March outcome is deliberately retained by the calling notebook.
Results remain exploratory until the separate outcome and temporal-validation repairs.
"""
from __future__ import annotations

import hashlib
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ['impressions_prior30', 'clicks_prior30', 'ctr_prior30', 'avg_position_prior30']
CATEGORICAL = ['content_type', 'main_intent']
FEATURES = NUMERIC + CATEGORICAL
KEYS = ['client_hash_id', 'content_hash_id']
TARGET = 'decoupling_signature'
CANDIDATES = ['Logistic Regression', 'Random Forest']
PRIMARY_K = 50


def make_model(name, seed=42, n_trees=300):
    # Each pipeline owns its transformer. Fitting one candidate cannot mutate another.
    prep = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    if name == 'Logistic Regression':
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f'Unknown candidate: {name}')
    return Pipeline([('prep', prep), ('clf', clf)])


def rank_metrics(scores, labels, ks=(25, 50, 100, 200)):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    if len(scores) != len(labels) or not len(scores):
        raise ValueError('Scores and labels must be nonempty and aligned.')
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Finite scores and both binary outcome classes are required.')
    if max(ks) > len(labels):
        raise ValueError('Not enough evaluation rows for every declared K.')
    # Caller canonically orders page keys. Stable ties follow those keys, never the label.
    order = np.argsort(-scores, kind='stable')
    prevalence = float(labels.mean())
    out = {'n_eval': len(labels), 'positive_count': int(labels.sum()),
           'base_rate': prevalence, 'roc_auc': float(roc_auc_score(labels, scores)),
           'average_precision': float(average_precision_score(labels, scores)),
           'unique_score_count': int(np.unique(scores).size),
           'zero_score_fraction': float((scores == 0).mean())}
    for k in ks:
        hits = int(labels[order[:k]].sum())
        cutoff = scores[order[k-1]]
        above = int((scores > cutoff).sum())
        at = scores == cutoff
        remaining = k-above
        tied_hits = int(labels[at].sum())
        tied_count = int(at.sum())
        above_hits = int(labels[scores > cutoff].sum())
        out.update({f'hits@{k}': hits, f'precision@{k}': hits/k,
                    f'recall@{k}': hits/int(labels.sum()),
                    f'lift@{k}': hits/k/prevalence,
                    f'cutoff_tie_count@{k}': tied_count,
                    # Expected performance under uniformly random cutoff tie breaking.
                    f'tie_expected_precision@{k}':
                        (above_hits + remaining*tied_hits/tied_count)/k,
                    f'tie_min_precision@{k}':
                        (above_hits + max(0, remaining-(tied_count-tied_hits)))/k,
                    f'tie_max_precision@{k}':
                        (above_hits + min(remaining, tied_hits))/k})
    return out


def choose_candidate(validation_table, primary_k=PRIMARY_K):
    """Fixed simplicity preference on exact ties; never use the outer test table."""
    learned = validation_table.set_index('method').loc[CANDIDATES]
    scores = learned[f'precision@{primary_k}']
    return next(name for name in CANDIDATES if scores[name] == scores.max())


def fit_reference(train, evaluation):
    """Original median-gap rule, fitted on training rows; diagnostics expose collapse."""
    bins = [0, 3, 10, 20, 50, np.inf]
    train_bucket = pd.cut(train.avg_position_prior30, bins, include_lowest=True)
    eval_bucket = pd.cut(evaluation.avg_position_prior30, bins, include_lowest=True)
    medians = train.groupby(train_bucket, observed=False).ctr_prior30.median()
    reference = eval_bucket.map(medians).astype(float)
    if reference.isna().any():
        raise ValueError('Baseline has an unsupported/missing position bucket; audit it.')
    visible = evaluation.impressions_prior30 >= train.impressions_prior30.median()
    return (visible * (reference-evaluation.ctr_prior30).clip(lower=0)).to_numpy()


def compare_models(frame, seed=42, n_trees=300, ks=(25, 50, 100, 200)):
    if PRIMARY_K not in ks:
        raise ValueError('The prespecified primary K=50 must be evaluated.')
    d = frame.sort_values(KEYS, kind='stable').reset_index(drop=True).copy()
    if d.duplicated(KEYS).any():
        raise ValueError('Page keys are not unique; audit the joins before fitting.')
    required = FEATURES + KEYS + [TARGET]
    if d[required].isna().any().any():
        raise ValueError('Missing required values; retain a documented cohort policy.')
    outer = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    outer_train_idx, test_idx = next(outer.split(d, groups=d.client_hash_id))
    outer_train, test = d.iloc[outer_train_idx].copy(), d.iloc[test_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed+1)
    inner_fit_idx, val_idx = next(inner.split(outer_train, groups=outer_train.client_hash_id))
    fit, validation = outer_train.iloc[inner_fit_idx], outer_train.iloc[val_idx]
    groups = [set(x.client_hash_id) for x in (fit, validation, test)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise AssertionError('Client overlap between fit, validation, and test.')
    for part in (fit, validation, test):
        if part[TARGET].nunique() != 2 or len(part) < max(ks):
            raise ValueError('Each split requires both classes and enough rows for K.')

    def evaluate(train, evaluation):
        models, scores = {}, {'Baseline rule': fit_reference(train, evaluation)}
        for name in CANDIDATES:
            model = make_model(name, seed=seed, n_trees=n_trees)
            model.fit(train[FEATURES], train[TARGET])
            models[name] = model
            scores[name] = model.predict_proba(evaluation[FEATURES])[:, 1]
        rows = [{'method': name, **rank_metrics(s, evaluation[TARGET], ks)}
                for name, s in scores.items()]
        return pd.DataFrame(rows), models, scores

    validation_table, _, _ = evaluate(fit, validation)
    selected = choose_candidate(validation_table)
    # Freeze selected before any outer test metrics are calculated.
    test_table, fitted, test_scores = evaluate(outer_train, test)
    membership = d[KEYS].copy()
    membership['split'] = 'inner_fit'
    membership.loc[outer_train.index[val_idx], 'split'] = 'inner_validation'
    membership.loc[test.index, 'split'] = 'outer_test'
    cohort_hash = hashlib.sha256(pd.util.hash_pandas_object(
        d[KEYS+FEATURES+[TARGET]], index=False).values.tobytes()).hexdigest()
    manifest = {
        'status': 'exploratory_stage_1_not_publication_validation',
        'primary_metric': 'precision@50',
        'selection_data': 'inner client validation only',
        'selection_tie_rule': 'prefer Logistic Regression',
        'ranking_tie_rule': 'canonical client/content key order, stable sort',
        'selected_method': selected, 'seed': seed, 'inner_seed': seed+1,
        'n_trees': n_trees, 'cohort_sha256': cohort_hash,
        'n_rows': len(d), 'n_clients': int(d.client_hash_id.nunique()),
        'partitions': {name: {'n_rows':len(part),
                             'n_clients':int(part.client_hash_id.nunique()),
                             'base_rate':float(part[TARGET].mean())}
                       for name,part in [('inner_fit',fit),('inner_validation',validation),
                                         ('outer_train',outer_train),('outer_test',test)]},
        'environment': {'python':platform.python_version(), 'numpy':np.__version__,
                        'pandas':pd.__version__, 'scikit_learn':sklearn.__version__},
        'known_pending_fixes': ['unequal calendar windows', 'zero-click eligibility',
                               'baseline validity', 'time-forward validation',
                               'client/time-aware uncertainty', 'prospective queue utility'],
        'note': 'Outer clients were examined in prior work; do not describe this as a new sealed test.'
    }
    return {'frame':d, 'validation':validation_table, 'test':test_table,
            'selected_method':selected, 'manifest':manifest, 'membership':membership,
            'test_frame':test, 'test_scores':test_scores, 'models':fitted}
