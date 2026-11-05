"""Descriptive support, overlap and client-quota checks on frozen existing scores.

These do not retrain models, choose new thresholds, or replace the global K=50 primary metric.
"""
import numpy as np
import pandas as pd

SUPPORT_THRESHOLDS=(1,5,20)
SECONDARY_CLIENT_QUOTA=10


def diagnostic_budget(scores,labels,k=50):
    s=np.asarray(scores,dtype=float);y=np.asarray(labels,dtype=int)
    if len(s)!=len(y) or not np.isfinite(s).all() or not set(np.unique(y)).issubset({0,1}):
        raise ValueError('Finite aligned scores and binary observed labels are required.')
    out={'n_pages':len(y),'positive_count':int(y.sum()),
         'base_rate':float(y.mean()) if len(y) else None,'budget':k}
    if len(y)<k:
        out.update(hits=None,precision=None,lift=None,reason='Insufficient pages for the declared budget; it was not silently reduced.')
        return out
    chosen=np.argsort(-s,kind='stable')[:k]
    hits=int(y[chosen].sum());precision=hits/k
    out.update(hits=hits,precision=precision,
               lift=precision/y.mean() if y.mean()>0 else None,reason=None)
    return out


def support_sensitivity(test,scores_by_method):
    rows=[]
    for threshold in SUPPORT_THRESHOLDS:
        mask=test.clicks_prior28.ge(threshold).to_numpy(dtype=bool)
        part=test.iloc[np.flatnonzero(mask)]
        for method,scores in scores_by_method.items():
            row=diagnostic_budget(np.asarray(scores)[mask],part.decoupling_signature)
            rows.append({'minimum_prior_clicks':threshold,'method':method,
                         'n_clients':int(part.client_hash_id.nunique()),**row})
    return rows


def ranking_overlap(test,scores_by_method,reference='Logistic Regression',k=50):
    if len(test)<k:raise ValueError('Overlap requires the full declared review budget.')
    y=test.decoupling_signature.to_numpy(dtype=int)
    reference_scores=np.asarray(scores_by_method[reference],dtype=float)
    a=set(np.argsort(-reference_scores,kind='stable')[:k].tolist())
    rank_a=pd.Series(reference_scores).rank(method='average').to_numpy(dtype=float)
    rows=[]
    for method,scores in scores_by_method.items():
        if method==reference:continue
        scores=np.asarray(scores,dtype=float)
        b=set(np.argsort(-scores,kind='stable')[:k].tolist())
        both=a&b;only_a=a-b;only_b=b-a
        rank_b=pd.Series(scores).rank(method='average').to_numpy(dtype=float)
        corr=float(np.corrcoef(rank_a,rank_b)[0,1]) if np.ptp(rank_a)>0 and np.ptp(rank_b)>0 else None
        rows.append({'reference':reference,'other_method':method,'budget':k,
            'overlap_pages':len(both),'jaccard':len(both)/len(a|b),
            'overlap_positive_count':int(y[list(both)].sum()) if both else 0,
            'reference_only_pages':len(only_a),
            'reference_only_positive_count':int(y[list(only_a)].sum()) if only_a else 0,
            'other_only_pages':len(only_b),
            'other_only_positive_count':int(y[list(only_b)].sum()) if only_b else 0,
            'spearman_rank_correlation_all_test_pages':corr})
    return rows


def client_quota_diagnostics(test,scores_by_method,quota=SECONDARY_CLIENT_QUOTA):
    if not isinstance(quota,int) or quota<1:raise ValueError('Quota must be a positive integer.')
    per_client=[];summary=[]
    client_array=test.client_hash_id.to_numpy()
    clients=sorted(test.client_hash_id.unique())
    for method,scores in scores_by_method.items():
        method_rows=[]
        for number,client in enumerate(clients,1):
            idx=np.flatnonzero(client_array==client)
            part=test.iloc[idx];s=np.asarray(scores,dtype=float)[idx]
            n=min(quota,len(part));chosen=np.argsort(-s,kind='stable')[:n]
            hits=int(part.decoupling_signature.to_numpy(dtype=int)[chosen].sum())
            row={'method':method,'anonymous_test_client':f'client_{number:02d}',
                 'available_pages':len(part),'positive_count':int(part.decoupling_signature.sum()),
                 'client_prevalence':float(part.decoupling_signature.mean()),
                 'reviewed_pages':n,'positive_hits':hits,'precision':hits/n}
            per_client.append(row);method_rows.append(row)
        reviewed=sum(r['reviewed_pages'] for r in method_rows)
        hits=sum(r['positive_hits'] for r in method_rows)
        summary.append({'method':method,'secondary_quota_per_client':quota,
            'n_clients':len(clients),'total_reviewed_pages':reviewed,'total_positive_hits':hits,
            'pooled_quota_precision':hits/reviewed,
            'macro_client_precision':float(np.mean([r['precision'] for r in method_rows])),
            'macro_client_prevalence':float(np.mean([r['client_prevalence'] for r in method_rows])),
            'all_clients_had_full_quota':all(r['reviewed_pages']==quota for r in method_rows)})
    return {'secondary_summary':summary,'anonymous_client_details':per_client}


def run_fixed_diagnostics(test,scores_by_method):
    if not len(test) or test.client_hash_id.isna().any():
        raise ValueError('Nonempty test data with client grouping keys are required.')
    y=test.decoupling_signature.to_numpy(dtype=int)
    for scores in scores_by_method.values():diagnostic_budget(scores,y)
    return {'support_sensitivity':support_sensitivity(test,scores_by_method),
            'ranking_overlap_against_logistic':ranking_overlap(test,scores_by_method),
            'client_quota_diagnostics':client_quota_diagnostics(test,scores_by_method),
            'interpretation':{
                'primary_global_budget':50,'primary_support_minimum_prior_clicks':1,
                'secondary_support_thresholds':[5,20],
                'fixed_models_not_refit':True,
                'quota_is_a_different_allocation_policy':True,
                'overlap_is_conditional_on_canonical_tie_policy':True,
                'no_threshold_selection_or_causal_claim':True,
                'five_client_and_single_period_limits_remain':True}}
