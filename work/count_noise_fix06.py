"""Descriptive conditional count-allocation audit; never a forecast feature."""
import hashlib,json
import numpy as np
import pandas as pd

MAX_TOTAL_CLICKS06=200
COUNT_PROTOCOL06={
    'version':'fix06-v1','status':'exploratory_reference_audit_after_Fix05_review',
    'purpose':'Describe raw event counts against a simple stationary click-allocation reference.',
    'source':'Existing Fix05 page-episode aggregates and unchanged cohorts; no new warehouse query or model fit.',
    'population':'Complete-case members of each fixed Fix05 arm, restricted to 1<=prior+future clicks<=200.',
    'cap_reason':'Predeclared computational scope for the low-total-count diagnostic; total clicks are conditioned on by the reference.',
    'null':'Independent Bernoulli clicks per impression with one common page CTR in the two windows.',
    'conditioning':'Both impression totals, the combined click total, and prior clicks>0.',
    'reference':'Future clicks X ~ Hypergeometric(M=I0+I1,n=C0+C1,N=I1), truncated to X<=C0+C1-1.',
    'signature_probability':'If stable impressions, F(floor(17*(C0+C1)/37))/F(C0+C1-1); otherwise 0.',
    'computation':'Finite-support hypergeometric weights from log-ratio recurrence; batches of at most 4096 rows.',
    'reporting':'Observed signatures beside sum of conditional probabilities on exactly the same modeled subset.',
    'prior_click_bands':'Observed counts only. No reference expectation is reported by observed prior-click band.',
    'not_forecast':'Reference uses future impressions and clicks. It is outcome analysis, never a predictor or ranking baseline.',
    'limitations':['The common-CTR independent-click model may fail under temporal dependence, changing query mix, seasonality or heterogeneous exposure.',
                   'Complete-case retention must be ignorable under this conditional reference; outcome-dependent observation can invalidate it.',
                   'Other eligibility/arm membership is treated as fixed under the reference.',
                   'Total-click restriction uses future counts and does not redefine the main prediction population.',
                   'No false-positive fraction, equivalence, calibrated hypothesis test, confidence interval or causal attribution follows.',
                   'Arms overlap and monthly pages/clients repeat; do not pool them as independent evidence.']}


def conditional_signature_probability06(i0,c0,i1,c1):
    arrays=[np.asarray(x,dtype=float) for x in [i0,c0,i1,c1]]
    if any(x.ndim!=1 for x in arrays) or len({len(x) for x in arrays})!=1:
        raise ValueError('Count arrays must be aligned one-dimensional arrays.')
    if any(not np.isfinite(x).all() or ((x<0)|(x>10**14)|(x!=np.floor(x))).any() for x in arrays):
        raise ValueError('Finite nonnegative integer counts within the audited range are required.')
    i0,c0,i1,c1=[x.astype('int64') for x in arrays]
    if ((i0<=0)|(c0<=0)|(c0>i0)|(c1>i1)).any():
        raise ValueError('Preserve positive-prior-click eligibility and clicks<=impressions.')
    total=c0+c1
    if (total>MAX_TOTAL_CLICKS06).any():raise ValueError('Apply the declared total-click scope before this reference.')
    stable=(9*i0<=10*i1)&(10*i1<=11*i0)
    q=np.zeros(len(total),dtype=float)
    if stable.any():
        selected=np.flatnonzero(stable)
        for start in range(0,len(selected),4096):
            ix=selected[start:start+4096];t=total[ix];a=i0[ix];b=i1[ix]
            lower=np.maximum(0,t-a);upper=np.minimum(b,t-1)
            if (lower>upper).any():raise ValueError('Prior-positive conditional support is empty.')
            width=int((upper-lower+1).max())
            log_weights=np.full((len(ix),width),-np.inf);log_weights[:,0]=0.
            for j in range(1,width):
                valid=lower+j<=upper;x=lower[valid]+j-1
                # w(x+1)/w(x) for C(b,x)*C(a,t-x); the unknown normalizer cancels.
                log_weights[valid,j]=(log_weights[valid,j-1]+np.log(b[valid]-x)
                    +np.log(t[valid]-x)-np.log(x+1)-np.log(a[valid]-t[valid]+x+1))
            weights=np.exp(log_weights-log_weights.max(axis=1,keepdims=True))
            numerator=(weights*((lower[:,None]+np.arange(width))<=((17*t)//37)[:,None])).sum(axis=1)
            values=numerator/weights.sum(axis=1)
            if not np.isfinite(values).all() or ((values<0)|(values>1+1e-12)).any():
                raise ValueError('Conditional probability calculation failed.')
            q[ix]=np.minimum(values,1.)
    return q


def observed_counts06(frame):
    known=frame.decoupling_signature.notna()
    positive=int(frame.loc[known,'decoupling_signature'].sum())
    return {'n_pages':len(frame),'n_observed':int(known.sum()),'n_unknown':int((~known).sum()),
            'observed_positive_count':positive,'observed_event_rate':positive/int(known.sum()) if known.any() else None}


def summarize_count_noise06(frame):
    observed=frame.loc[frame.decoupling_signature.notna()].copy()
    fields=['impressions_prior28','clicks_prior28','impressions_outcome','clicks_outcome']
    counts=observed[fields].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or ((counts<0)|(counts>10**14)|(counts!=np.floor(counts))).any():
        raise ValueError('Invalid complete-case counts.')
    i0,c0,i1,c1=counts.astype('int64').T
    if ((i0<=0)|(c0<=0)|(c0>i0)|(c1>i1)).any():raise ValueError('Invalid click/impression counts.')
    actual=((9*i0<=10*i1)&(10*i1<=11*i0)&(20*c1<=17*c0)).astype(int)
    if not np.array_equal(actual,observed.decoupling_signature.to_numpy(dtype=int)):
        raise ValueError('Observed labels differ from the frozen integer rule.')
    in_scope=(c0+c1)<=MAX_TOTAL_CLICKS06
    modeled=observed.loc[in_scope].copy()
    q=conditional_signature_probability06(i0[in_scope],c0[in_scope],i1[in_scope],c1[in_scope])
    count=int(modeled.decoupling_signature.sum());expected=float(q.sum());n=len(modeled)
    bands=[]
    for name,low,high in [('1-4',1,5),('5-19',5,20),('20+',20,np.inf)]:
        part=frame.loc[frame.clicks_prior28.ge(low)&frame.clicks_prior28.lt(high)]
        bands.append({'prior_click_band':name,**observed_counts06(part)})
    return {'full_arm_observation':observed_counts06(frame),
        'reference_subset':{'n_modeled_pages':n,'n_observed_pages_excluded_total_clicks_gt_200':int((~in_scope).sum()),
            'fraction_of_observed_pages_modeled':n/len(observed) if len(observed) else None,
            'n_stable_impression_pages':int(((9*i0[in_scope]<=10*i1[in_scope])&(10*i1[in_scope]<=11*i0[in_scope])).sum()),
            'observed_signature_count':count,'stationary_reference_expected_signature_count':expected,
            'observed_minus_reference_expected_count_descriptive':count-expected,
            'observed_signature_rate':count/n if n else None,
            'stationary_reference_expected_signature_rate':expected/n if n else None},
        'prior_click_bands_observed_only':bands,
        'reference_input_sha256':hashlib.sha256(pd.util.hash_pandas_object(observed[fields+['decoupling_signature']],index=False).values.tobytes()).hexdigest()}


def audit_fixed_episodes06(episodes,validation):
    rows=[]
    keys=['client_hash_id','content_hash_id']
    features=['impressions_prior28','clicks_prior28','ctr_prior28','avg_position_prior28']
    for fold in validation['folds']:
        origin=next(x for x in episodes if str(x)==fold['origin'])
        train_clients={client for t,frame in episodes.items() if str(t) in fold['training_origins']
                       for client in frame.loc[frame.decoupling_signature.notna(),'client_hash_id']}
        future=episodes[origin]
        for arm in fold['arms']:
            if arm['status']!='scored':continue
            if arm['arm']=='hash_heldout_clients':
                mask=future.client_hash_id.map(lambda client:int(hashlib.sha256(('flyrank-fix05-v1|'+str(client)).encode()).hexdigest(),16)%4==0)
            else:
                seen=future.client_hash_id.isin(train_clients)
                mask=seen if arm['arm']=='known_clients' else ~seen
            frame=future.loc[mask].sort_values(keys,kind='stable').reset_index(drop=True)
            digest=hashlib.sha256(pd.util.hash_pandas_object(frame[keys+features+['decoupling_signature']],index=False).values.tobytes()).hexdigest()
            if digest!=arm['cohort_sha256']:raise ValueError('Fix05 cohort/features/labels differ from the returned run.')
            stats=summarize_count_noise06(frame)
            if stats['full_arm_observation']['n_pages']!=arm['evaluation']['n_page_episodes']:
                raise ValueError('Evaluation population changed.')
            rows.append({'origin':str(origin),'arm':arm['arm'],'source_cohort_sha256':digest,**stats})
    return {'protocol':COUNT_PROTOCOL06,'protocol_sha256':hashlib.sha256(json.dumps(COUNT_PROTOCOL06,sort_keys=True).encode()).hexdigest(),
            'source_fix05_protocol_sha256':validation['protocol_sha256'],'rows':rows,
            'interpretation':'Reference-model diagnostic only. An observed signature is not declared spurious, and model rankings are unchanged.'}
