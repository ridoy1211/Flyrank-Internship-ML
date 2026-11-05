"""Frozen retrospective time-ordered evaluation; no warehouse access on import."""
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
from .baselines import fit_pooled_reference, score_transparent_baselines

KEYS05=['client_hash_id','content_hash_id']
FEATURES05=['impressions_prior28','clicks_prior28','ctr_prior28','avg_position_prior28']
METHODS05=['Baseline rule','Logistic Regression','Random Forest','Prior clicks',
           'Prior impressions','Prior CTR','Visible pooled-CTR shortfall','Visible pooled-CTR excess']
TEST_ORIGINS05=[date(2026,m,1) for m in (2,3,4,5)]
EPISODES05=[date(2025,11,1),date(2025,12,1)]+[date(2026,m,1) for m in range(1,6)]
MONTHS05=['2025-10','2025-11','2025-12']+[f'2026-{m:02d}' for m in range(1,6)]
PROTOCOL05={
    'version':'fix05-v1', 'status':'frozen_retrospective_protocol_not_preregistration',
    'question':'Does a fixed learned ranking add retrieval value beyond simple rules on later periods?',
    'primary_learned_reference':'Logistic Regression','methods':METHODS05,
    'features':FEATURES05,'metadata_used':False,'ids_are_features':False,
    'origins':[str(t) for t in TEST_ORIGINS05],
    'development_origins':['2026-02-01','2026-03-01'],
    'later_origins':['2026-04-01','2026-05-01'],
    'later_origin_status':'Not scored by this repair workflow; prior user exposure is unknown. Not a blind-test claim.',
    'feature_window':'t-31 through t-4 inclusive, 28 days',
    'outcome_window':'t through t+27 inclusive, 28 days',
    'assumed_gap':'Three full date labels t-3,t-2,t-1; all training outcome dates must also be <=t-4.',
    'availability_limitation':'Retrospective final release; no historical ingestion vintages. Buffer is an assumption, not verified availability.',
    'date_convention':'Warehouse report_date labels, without timezone conversion; source timezone not independently verified.',
    'training':'Exactly the last three mature episode origins, pooled page-episodes; no shorter fallback history.',
    'training_guards':{'n_labelable_rows':500,'n_clients':2,'each_class':20},
    'primary_population':'As-of eligible pages of clients with a labelable row in the rolling training frame.',
    'eligibility':'28 explicit available prior daily rows; I>0, C>0, finite positive impression-weighted position. Sparse/missing days are not zero-filled.',
    'outcome_observability':'28 explicit available outcome rows with valid counts; otherwise unknown, retained in the as-of queue.',
    'label':'9*I_prior<=10*I_outcome<=11*I_prior and 20*C_outcome<=17*C_prior',
    'primary_budget':50,'secondary_budgets':[25,100,200],
    'primary_reporting':'Rank all eligible pages before outcome filtering. Report observed hits, unknowns, bounds; point precision only for fully observed queues.',
    'secondary_reporting':'Complete-case metrics, support >=5/20 without refitting, anonymous client contributions.',
    'tie_policy':'Descending score, stable canonical client/content key order; conditional tie ranges also reported.',
    'hash_holdout':'sha256(flyrank-fix05-v1|client_id) mod 4 == 0; excluded from all hash-arm training; descriptive.',
    'models':{'LR':{'C':1.0,'max_iter':1000,'seed':42},
              'RF':{'n_estimators':300,'seed':42,'n_jobs':-1,'other_parameters':'sklearn 1.7.2 defaults'},
              'preprocessing':'Independent StandardScaler per model, four numeric features only; no tuning or winner selection.'},
    'rules':'Training-only position bands (0,3,10,20,50,infinity); medians or pooled CTR; visibility at training median I. High prior counts/CTR rank first.',
    'unseen_band_fallback':'Global training median CTR for original rule; global pooled training CTR for pooled rules.',
    'source_months':MONTHS05,'final_month':'June 2026 is excluded, including the sample table.',
    'inference_limits':'Repeated pages and clients; no iid page confidence intervals. Few periods/clients may preclude a strong claim.',
    'changes_from_stage2':'Numeric-only features, lagged schedule, rolling fitting, different client split and as-of queues. Changes cannot be attributed solely to time splitting.'}


def protocol_hash05():
    return hashlib.sha256(json.dumps(PROTOCOL05,sort_keys=True).encode()).hexdigest()


def windows05():
    rows=[]
    for t in EPISODES05:
        for role,start,end in [('feature',t-timedelta(days=31),t-timedelta(days=4)),
                               ('outcome',t,t+timedelta(days=27))]:
            rows.append({'origin':str(t),'role':role,'start_date':start,'end_date':end})
    return pd.DataFrame(rows)


def train_origins05(t):
    return [s for s in EPISODES05 if s<t and s+timedelta(days=27)<=t-timedelta(days=4)][-3:]


def heldout05(client):
    return int(hashlib.sha256(('flyrank-fix05-v1|'+str(client)).encode()).hexdigest(),16)%4==0


def build_episode05(aggregate,origin):
    a=aggregate.copy()
    if a.duplicated(['role']+KEYS05).any() or a[KEYS05].isna().any().any():
        raise ValueError('Invalid aggregate grain or missing page keys.')
    if (a.raw_rows!=a.raw_days).any():
        raise ValueError('Duplicate daily keys; stop before modelling.')
    if (a.bad_counts>0).any():
        raise ValueError('Invalid available count data; stop before modelling.')
    for column in ['impressions','clicks']:
        values=a[column].dropna().to_numpy(dtype=float)
        if ((values<0)|(values>10**14)|(values!=np.floor(values))|~np.isfinite(values)).any():
            raise ValueError('Count range/integer guard failed.')
    f=a.loc[a.role.eq('feature')].copy()
    o=a.loc[a.role.eq('outcome'),KEYS05+['available_days','impressions','clicks']].copy()
    o=o.rename(columns={c:c+'_outcome' for c in ['available_days','impressions','clicks']})
    position=f.sum_position/f.impressions.replace(0,np.nan)
    full=f.available_days.eq(28)
    positive=full&f.impressions.gt(0)&f.clicks.gt(0)
    eligible=positive&f.bad_position.eq(0)&np.isfinite(position)&position.gt(0)
    flow=[]
    for name,mask in [('all_prior_pages',pd.Series(True,index=f.index)),
                      ('complete_prior_rows',full),('positive_prior_counts',positive),
                      ('as_of_eligible',eligible)]:
        flow.append({'stage':name,'n_pages':int(mask.sum()),'n_clients':int(f.loc[mask,'client_hash_id'].nunique())})
    f=f.loc[eligible].copy()
    f['avg_position_prior28']=position.loc[eligible]
    f=f.rename(columns={'impressions':'impressions_prior28','clicks':'clicks_prior28'})
    f['ctr_prior28']=100*f.clicks_prior28/f.impressions_prior28
    d=f[KEYS05+FEATURES05].merge(o,on=KEYS05,how='left',validate='one_to_one')
    observed=d.available_days_outcome.eq(28)
    for col in ['impressions_prior28','clicks_prior28','impressions_outcome','clicks_outcome']:
        d[col]=d[col].astype('Int64')
    label=((9*d.impressions_prior28<=10*d.impressions_outcome)&
           (10*d.impressions_outcome<=11*d.impressions_prior28)&
           (20*d.clicks_outcome<=17*d.clicks_prior28)).astype('Int64')
    d['decoupling_signature']=label.where(observed,pd.NA)
    d['origin']=str(origin)
    d=d.sort_values(KEYS05,kind='stable').reset_index(drop=True)
    return d,{'origin':str(origin),'flow':flow,'n_labelable':int(observed.sum()),
              'n_unknown':int((~observed).sum()),'n_positive':int(d.decoupling_signature.sum())}


def score_models05(train,evaluation):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    # No outcome or identifier column is passed to either model.
    scores={}
    bucket=pd.cut(train.avg_position_prior28,[0,3,10,20,50,np.inf],include_lowest=True)
    median=train.groupby(bucket,observed=False).ctr_prior28.median()
    e_bucket=pd.cut(evaluation.avg_position_prior28,[0,3,10,20,50,np.inf],include_lowest=True)
    expected=e_bucket.map(median).astype(float).fillna(float(train.ctr_prior28.median()))
    scores['Baseline rule']=((evaluation.impressions_prior28>=train.impressions_prior28.median())*
                            (expected-evaluation.ctr_prior28).clip(lower=0)).to_numpy(dtype=float)
    for name,clf in [('Logistic Regression',LogisticRegression(C=1.0,max_iter=1000,random_state=42)),
                     ('Random Forest',RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1))]:
        model=make_pipeline(StandardScaler(),clf)
        with warnings.catch_warnings(),threadpool_limits(limits=1):
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(train[FEATURES05].to_numpy(dtype=float),train.decoupling_signature.to_numpy(dtype=int))
            scores[name]=model.predict_proba(evaluation[FEATURES05].to_numpy(dtype=float))[:,1]
    rules,audit=score_transparent_baselines(fit_pooled_reference(train),evaluation)
    return {**scores,**rules},audit


def metrics05(frame,scores,ks=(25,50,100,200)):
    from sklearn.metrics import roc_auc_score,average_precision_score
    s=np.asarray(scores,dtype=float)
    if len(s)!=len(frame) or not np.isfinite(s).all():
        raise ValueError('Non-finite or unaligned predictions.')
    known=frame.decoupling_signature.notna().to_numpy()
    y=frame.decoupling_signature.fillna(0).to_numpy(dtype=int)
    order=np.argsort(-s,kind='stable')
    complete_order=np.flatnonzero(known)[np.argsort(-s[known],kind='stable')]
    positives=int(y[known].sum()); n=int(known.sum())
    base=float(positives/n) if n else None
    out={'n_as_of':len(frame),'n_observed':n,'n_unknown':len(frame)-n,
         'observed_positive_count':positives,'complete_case_prevalence':base,
         'roc_auc_complete_case':float(roc_auc_score(y[known],s[known])) if n and np.unique(y[known]).size==2 else None,
         'average_precision_complete_case':float(average_precision_score(y[known],s[known])) if positives else None,
         'queues':[],'complete_case_rank_metrics':[]}
    for k in ks:
        if len(s)<k:
            queue={'budget':k,'reason':'insufficient_as_of_pages','reviewed':None}
        else:
            ix=order[:k]; hits=int(y[ix].sum()); unknown=int((~known[ix]).sum())
            cutoff=s[ix[-1]]; above=s>cutoff; tied=s==cutoff
            slots=k-int(above.sum()); tie_n=int(tied.sum()); tie_hits=int(y[tied].sum())
            tie_unknown=int((tied&~known).sum()); above_hits=int(y[above].sum())
            above_unknown=int((above&~known).sum())
            queue={'budget':k,'reason':None,'reviewed':k,'observed_hits':hits,'unknown_outcomes':unknown,
                   'label_coverage':(k-unknown)/k,'precision_if_complete':hits/k if unknown==0 else None,
                   'precision_lower_bound':hits/k,'precision_upper_bound':(hits+unknown)/k,
                   'cutoff_tie_count':tie_n,
                   'tie_and_missing_lower_bound':(above_hits+max(0,slots-(tie_n-tie_hits)))/k,
                   'tie_and_missing_upper_bound':(above_hits+above_unknown+min(slots,tie_hits+tie_unknown))/k}
        out['queues'].append(queue)
        if n<k:
            cc={'budget':k,'precision':None,'reason':'insufficient_observed_pages'}
        else:
            hits=int(y[complete_order[:k]].sum())
            cc={'budget':k,'hits':hits,'precision':hits/k,'lift':hits/k/base if base else None,'reason':None}
        out['complete_case_rank_metrics'].append(cc)
    return out


def cohort_summary05(frame):
    n=len(frame); known=frame.decoupling_signature.notna()
    sizes=frame.groupby('client_hash_id').size().sort_values(ascending=False)
    return {'n_page_episodes':n,'n_clients':len(sizes),'n_labelable':int(known.sum()),
            'n_positive':int(frame.decoupling_signature.sum()),
            'n_clients_with_at_least_50_pages':int((sizes>=50).sum()),
            'largest_two_client_fraction':float(sizes.iloc[:2].sum()/n) if n else None}


def run_chronological05(episodes,cohort_audits,score_sink=None,progress=None):
    results=[]
    # Names are export-only aliases, never predictors or split inputs.
    roster=sorted({str(x) for frame in episodes.values() for x in frame.client_hash_id})
    aliases={client:f'client_{i+1:03d}' for i,client in enumerate(roster)}
    for t in TEST_ORIGINS05:
        origins=train_origins05(t)
        fold={'origin':str(t),'cutoff':str(t-timedelta(days=4)),
              'training_origins':[str(s) for s in origins],'arms':[]}
        results.append(fold)
        if len(origins)!=3 or any(s not in episodes or not episodes[s].decoupling_signature.notna().any() for s in origins):
            fold['status']='insufficient_history';continue
        train=pd.concat([episodes[s] for s in origins],ignore_index=True)
        train=train.loc[train.decoupling_signature.notna()].copy()
        future=episodes[t].copy()
        for arm in ['known_clients','naturally_unseen_clients','hash_heldout_clients']:
            tr=train.copy()
            if arm=='hash_heldout_clients':
                tr=tr.loc[~tr.client_hash_id.map(heldout05)]
                test=future.loc[future.client_hash_id.map(heldout05)].copy()
            else:
                seen=future.client_hash_id.isin(set(tr.client_hash_id))
                test=future.loc[seen if arm=='known_clients' else ~seen].copy()
            test=test.sort_values(KEYS05,kind='stable').reset_index(drop=True)
            record={'arm':arm,'training':cohort_summary05(tr),'evaluation':cohort_summary05(test),'metrics':[]}
            fold['arms'].append(record)
            if set(tr.origin)!={str(s) for s in origins}:
                record['status']='insufficient_training_history';continue
            counts=tr.decoupling_signature.value_counts()
            if len(tr)<500 or tr.client_hash_id.nunique()<2 or min(counts.get(0,0),counts.get(1,0))<20:
                record['status']='insufficient_training_support';continue
            if not len(test):
                record['status']='no_eligible_test_pages';continue
            if arm=='hash_heldout_clients' and set(tr.client_hash_id)&set(test.client_hash_id):
                raise AssertionError('Hash holdout client leakage.')
            if progress is not None:
                progress(f'Fitting {t} {arm}')
            scores,reference_audit=score_models05(tr,test)
            if score_sink is not None:
                score_sink(str(t),arm,test,scores)
            record['status']='scored';record['reference_audit']=reference_audit
            record['prediction_sha256']={name:hashlib.sha256(np.asarray(s,dtype=float).tobytes()).hexdigest()
                                          for name,s in scores.items()}
            record['cohort_sha256']=hashlib.sha256(pd.util.hash_pandas_object(
                test[KEYS05+FEATURES05+['decoupling_signature']],index=False).values.tobytes()).hexdigest()
            for minimum in (1,5,20):
                mask=test.clicks_prior28.ge(minimum).to_numpy(dtype=bool)
                for name,s in scores.items():
                    record['metrics'].append({'method':name,'minimum_prior_clicks':minimum,
                                               'support_population':cohort_summary05(test.loc[mask]),
                                               **metrics05(test.loc[mask],s[mask])})
            details=[]
            global_orders={name:np.argsort(-s,kind='stable')[:50] if len(test)>=50 else np.array([],dtype=int)
                           for name,s in scores.items()}
            for client,part in test.groupby('client_hash_id',sort=True):
                for name,order in global_orders.items():
                    selected=test.iloc[order]; selected=selected.loc[selected.client_hash_id.eq(client)]
                    details.append({'anonymous_client':aliases[str(client)],'method':name,
                        **cohort_summary05(part),'global_queue_pages':len(selected),
                        'global_queue_observed_hits':int(selected.decoupling_signature.sum()),
                        'global_queue_unknown':int(selected.decoupling_signature.isna().sum())})
            record['anonymous_client_details']=details
        fold['status']='completed_with_explicit_arm_statuses'
    return {'protocol':PROTOCOL05,'protocol_sha256':protocol_hash05(),
            'cohort_audits':cohort_audits,'folds':results,
            'limitations':['Retrospective final release, assumed availability buffer.',
                           'Outcome missingness is not assumed random; main queue bounds are descriptive.',
                           'Repeated page-episodes and few clients/periods; no iid uncertainty claims.',
                           'No causal attribution or evidence that editing pages improves outcomes.']}
