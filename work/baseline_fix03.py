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


def queue_profile(evaluation,scores,k=50):
    """Aggregate error/support profile; hashed IDs are used only for grouping."""
    order=np.argsort(-np.asarray(scores,dtype=float),kind='stable')[:k]
    part=evaluation.iloc[order]
    counts=part.client_hash_id.value_counts()
    positive=part.decoupling_signature.eq(1)
    return {'n_reviewed':len(part),'n_label_positive':int(positive.sum()),
            'n_clients_represented':int(part.client_hash_id.nunique()),
            'largest_client_fraction':float(counts.max()/len(part)),
            'n_clients_with_positive_hits':int(part.loc[positive,'client_hash_id'].nunique()),
            'median_prior_clicks':float(part.clicks_prior28.median()),
            'median_prior_impressions':float(part.impressions_prior28.median()),
            'median_prior_ctr':float(part.ctr_prior28.median()),
            'median_prior_position':float(part.avg_position_prior28.median()),
            'manual_page_review_performed':False}


def evaluate_fixed_stage2(result,rank_metrics):
    """Preserve model selection and all existing learned-model scores and partitions."""
    manifest=result['manifest']
    if manifest.get('status')!='exploratory_stage_2_measurement_repair':
        raise ValueError('Run the complete Fix 02 notebook in this runtime first.')
    if manifest.get('cohort_sha256')!='66e700f61bfabcac440f1383bee3c402a75c15b700ba33e77b0887a8741d3560':
        raise ValueError('The Stage 2 cohort differs from the reviewed run; review the change before comparing.')
    d=result['frame'];membership=result['membership']
    keys=['client_hash_id','content_hash_id']
    hash_columns=keys+BASELINE_FEATURES+['content_type','main_intent','decoupling_signature']
    current_hash=hashlib.sha256(pd.util.hash_pandas_object(d[hash_columns],index=False).values.tobytes()).hexdigest()
    if current_hash!=manifest['cohort_sha256']:
        raise ValueError('Stage 2 features or labels changed after the recorded run; preserve the reviewed cohort.')
    if not d[keys].equals(membership[keys]):
        raise ValueError('Split membership is not aligned to the canonical frame.')
    fit=d.loc[membership.split.eq('inner_fit')]
    validation=d.loc[membership.split.eq('inner_validation')]
    outer_train=d.loc[~membership.split.eq('outer_test')]
    test=d.loc[membership.split.eq('outer_test')]
    if not test[keys].equals(result['test_frame'][keys]):
        raise ValueError('Outer test rows/order have changed.')
    tables={};diagnostics={};profiles={}
    for name,train,evaluation,old in [
        ('validation',fit,validation,result['validation']),
        ('outer_test',outer_train,test,result['test'])]:
        reference=fit_pooled_reference(train)
        scores,audit=score_transparent_baselines(reference,evaluation)
        rows=[{'method':method,**rank_metrics(score,evaluation.decoupling_signature)}
              for method,score in scores.items()]
        tables[name]=pd.concat([old.copy(),pd.DataFrame(rows)],ignore_index=True)
        diagnostics[name]=audit
        if name=='outer_test':
            profiles={method:queue_profile(evaluation,score)
                      for method,score in {**result['test_scores'],**scores}.items()}
    return {'tables':tables,'diagnostics':diagnostics,'outer_test_queue_profiles':profiles,
            'selected_learned_candidate_unchanged':result['selected_method'],
            'interpretation':'Exploratory comparator audit on an already examined client split; freeze before future-period testing.'}
