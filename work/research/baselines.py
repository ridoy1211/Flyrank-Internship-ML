"""Transparent, prespecified comparators for the fixed Stage 2 outcome.

All references use training features only. Scores are priorities, not probabilities,
causes, or claims that editing a page will recover clicks.
"""
import hashlib
import numpy as np
import pandas as pd

BASELINE_FEATURES=['impressions_prior28','clicks_prior28','ctr_prior28','avg_position_prior28']
NEW_BASELINES=['Prior clicks','Prior impressions','Prior CTR',
               'Visible pooled-CTR shortfall','Visible pooled-CTR excess']


def validate_baseline_features(frame):
    x=frame[BASELINE_FEATURES].to_numpy(dtype=float)
    if not len(frame) or not np.isfinite(x).all():
        raise ValueError('Baseline input must be nonempty with finite prior-window features.')
    if (frame.impressions_prior28<=0).any() or (frame.clicks_prior28<=0).any():
        raise ValueError('Keep the fixed Stage 2 positive-count eligibility policy.')
    if (frame.clicks_prior28>frame.impressions_prior28).any():
        raise ValueError('Clicks exceed impressions; audit measurement before scoring.')
    if (frame.avg_position_prior28<=0).any():
        raise ValueError('Stage 2 requires positive observed positions.')
    if not np.allclose(x[:,2],100*x[:,1]/x[:,0]):
        raise ValueError('Prior CTR does not match prior clicks/impressions on the 0–100 scale.')


def fit_pooled_reference(train):
    validate_baseline_features(train)
    bucket=pd.cut(train.avg_position_prior28,[0,3,10,20,50,np.inf],include_lowest=True)
    totals=train.groupby(bucket,observed=False).agg(
        impressions=('impressions_prior28','sum'),clicks=('clicks_prior28','sum'),
        n_pages=('impressions_prior28','size'))
    rates=100*totals.clicks/totals.impressions.replace(0,np.nan)
    return {'rates':rates,
            'global_ctr':float(100*train.clicks_prior28.sum()/train.impressions_prior28.sum()),
            'visibility_threshold':float(train.impressions_prior28.median()),
            'training_bucket_support':[{'position_bucket':str(index),
                'n_pages':int(row.n_pages),'impressions':float(row.impressions),
                'clicks':float(row.clicks),
                'pooled_ctr':float(rates.loc[index]) if pd.notna(rates.loc[index]) else None}
                for index,row in totals.iterrows()]}


def score_transparent_baselines(reference,evaluation):
    validate_baseline_features(evaluation)
    bucket=pd.cut(evaluation.avg_position_prior28,[0,3,10,20,50,np.inf],include_lowest=True)
    expected=bucket.map(reference['rates']).astype(float)
    fallback=expected.isna()
    expected=expected.fillna(reference['global_ctr'])
    visible=(evaluation.impressions_prior28>=reference['visibility_threshold']).astype(int)
    scores={
        'Prior clicks':evaluation.clicks_prior28.to_numpy(dtype=float),
        'Prior impressions':evaluation.impressions_prior28.to_numpy(dtype=float),
        'Prior CTR':evaluation.ctr_prior28.to_numpy(dtype=float),
        'Visible pooled-CTR shortfall':(visible*(expected-evaluation.ctr_prior28).clip(lower=0)).to_numpy(dtype=float),
        'Visible pooled-CTR excess':(visible*(evaluation.ctr_prior28-expected).clip(lower=0)).to_numpy(dtype=float),
    }
    audit={'n_evaluation_pages':len(evaluation),
           'n_unseen_bucket_global_fallback':int(fallback.sum()),
           'training_global_ctr':reference['global_ctr'],
           'training_visibility_threshold':reference['visibility_threshold'],
           'training_bucket_support':reference['training_bucket_support']}
    return scores,audit
