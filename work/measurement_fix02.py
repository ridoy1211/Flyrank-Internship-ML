"""Equal-window event measurement; outcome availability is distinct from a negative label."""
import numpy as np
import pandas as pd

KEYS = ['client_hash_id', 'content_hash_id']


def repaired_label(prior_impressions, prior_clicks, outcome_impressions, outcome_clicks, observed):
    """Exact integer threshold comparisons; undefined or unobserved events stay NA."""
    for series in [prior_impressions,prior_clicks,outcome_impressions,outcome_clicks]:
        values=series.dropna()
        if not np.isfinite(values).all() or (values<0).any() or (values%1!=0).any():
            raise ValueError('Count metrics must be finite, nonnegative integers when present.')
    index=prior_impressions.index
    result=pd.Series(pd.NA,index=index,dtype='Int64')
    known=(observed & prior_impressions.gt(0) & prior_clicks.gt(0)
           & outcome_impressions.notna() & outcome_clicks.notna())
    pi=prior_impressions.loc[known].astype('int64')
    pc=prior_clicks.loc[known].astype('int64')
    oi=outcome_impressions.loc[known].astype('int64')
    oc=outcome_clicks.loc[known].astype('int64')
    result.loc[known]=((10*oi>=9*pi)&(10*oi<=11*pi)&(20*oc<=17*pc)).astype('int64')
    return result


def prepare_equal_windows(feb, mar, metadata):
    """Inputs are page-level aggregates with counts of observed and available dates."""
    for name,frame in [('February',feb),('March',mar),('metadata',metadata)]:
        if frame.duplicated(KEYS).any():
            raise ValueError(f'{name}: duplicate page keys; audit before merging.')
    for name,frame in [('February',feb),('March',mar)]:
        if (frame.raw_rows != frame.raw_days).any():
            raise ValueError(f'{name}: duplicate page-day rows; do not sum them twice.')
        if (frame.bad_counts > 0).any():
            raise ValueError(f'{name}: available rows with null/negative count metrics.')
    d=feb.merge(metadata,on=KEYS,how='left',validate='one_to_one')
    # A missing outcome is missing; counts are never replaced with zero here.
    d=d.merge(mar.drop(columns=['raw_rows','raw_days','bad_counts']),on=KEYS,
              how='left',validate='one_to_one')
    full_feb=d.available_days_prior28.eq(28)
    positive_impressions=d.impressions_prior28.gt(0)
    positive_clicks=d.clicks_prior28.gt(0)
    supported_metadata=d[['content_type','main_intent']].notna().all(axis=1)
    supported_position=d.avg_position_prior28.notna()&d.avg_position_prior28.gt(0)
    stages=[('all_february_pages',pd.Series(True,index=d.index)),
            ('complete_february_coverage',full_feb),
            ('positive_prior_impressions',full_feb&positive_impressions),
            ('positive_prior_clicks',full_feb&positive_impressions&positive_clicks),
            ('required_features_available',full_feb&positive_impressions&positive_clicks
                                          &supported_metadata&supported_position)]
    flow=[{'stage':name,'n_pages':int(mask.sum()),
           'n_clients':int(d.loc[mask,'client_hash_id'].nunique())} for name,mask in stages]
    eligible=stages[-1][1]
    observed28=d.available_days_outcome28.eq(28)
    observed31=d.available_days_outcome31.eq(31)
    d['ctr_prior28']=100*d.clicks_prior28/d.impressions_prior28.replace(0,np.nan)
    d['decoupling_signature']=repaired_label(d.impressions_prior28,d.clicks_prior28,
        d.impressions_outcome28,d.clicks_outcome28,eligible&observed28)
    d['calendar31_control_label']=repaired_label(d.impressions_prior28,d.clicks_prior28,
        d.impressions_outcome31,d.clicks_outcome31,eligible&observed31)
    evaluation=d.loc[eligible&observed28].copy()
    evaluation['decoupling_signature']=evaluation.decoupling_signature.astype('int64')
    paired=d.loc[eligible&observed28&observed31]
    label_table=[]
    for a in (0,1):
        for b in (0,1):
            label_table.append({'equal28_label':a,'unequal31_label':b,
                'n_pages':int(((paired.decoupling_signature==a)&
                               (paired.calendar31_control_label==b)).sum())})
    buckets=pd.cut(evaluation.clicks_prior28,bins=[0,4,19,np.inf],
                   labels=['1-4','5-19','20+'])
    support=[]
    for band in ['1-4','5-19','20+']:
        part=evaluation.loc[buckets==band]
        support.append({'prior_click_band':band,'n_pages':len(part),
                        'positive_count':int(part.decoupling_signature.sum()),
                        'base_rate':float(part.decoupling_signature.mean()) if len(part) else None})
    audit={'feature_window':'2026-02-01 through 2026-02-28 inclusive',
        'outcome_window':'2026-03-01 through 2026-03-28 inclusive',
        'calendar_control_window':'2026-03-01 through 2026-03-31 inclusive',
        'as_of_eligibility':'28 available prior days; prior impressions>0 and clicks>0; required features present; positive position',
        'outcome_observability':'28 available outcome days; missing outcome is not a negative',
        'flow':flow,'n_as_of_eligible':int(eligible.sum()),
        'n_eligible_without_complete_outcome':int((eligible&~observed28).sum()),
        'n_evaluation_pages':len(evaluation),
        'n_zero_prior_clicks_with_full_prior_exposure':int((full_feb&positive_impressions&d.clicks_prior28.eq(0)).sum()),
        'paired_calendar_comparison_n':len(paired),'paired_label_counts':label_table,
        'prior_click_support':support,
        'measurement_note':'Low positive counts are retained in the main task and reported by support band; they are not claimed reliable business losses.',
        'scope_note':'Evaluation conditions on complete outcome observation; coverage losses are disclosed, not assumed random.'}
    return evaluation,audit
