"""Fixed-queue partial identification and client-portfolio influence diagnostics."""
import hashlib,json,warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from baseline_fix03 import fit_pooled_reference,score_transparent_baselines

KEYS07=['client_hash_id','content_hash_id']
FEATURES07=['impressions_prior28','clicks_prior28','ctr_prior28','avg_position_prior28']
PAIRS07=[('Logistic Regression','Prior CTR'),('Logistic Regression','Visible pooled-CTR excess')]
PROTOCOL07={
    'version':'fix07-v1','status':'exploratory_paired_diagnostic_after_Fix06_review',
    'population':'Unchanged Fix05 known-client as-of eligible frames; no total-click cap or future-outcome filtering.',
    'origins':['2026-02-01','2026-03-01','2026-04-01','2026-05-01'],
    'budget':50,'pairs':[list(pair) for pair in PAIRS07],
    'replay':'Four fits of unchanged LR settings recover the original predictions. Require exact prediction hashes for LR, prior CTR and pooled CTR excess.',
    'models':'No new candidate, tuning or Random-Forest fit. All eight original Fix05 results remain retained.',
    'tie_policy':'Stable descending scores in original canonical client/content order.',
    'pair_bound':'Let d be known-positive A-only minus B-only. Bounds on P50(A)-P50(B) are (d-unknown_B_only)/50 and (d+unknown_A_only)/50.',
    'overlap':'Shared pages, including shared unknown outcomes, cancel exactly in a paired difference.',
    'client_influence':'Delete each eligible client once, rerank fixed predictions on remaining pages at the same K. Do not refit after deletion.',
    'limitations':['Missing-data bounds apply to these finite queues under the frozen tie policy; they are not confidence intervals.',
                   'Client deletion measures evaluation-portfolio sensitivity, not uncertainty of training or unseen-client model performance.',
                   'Repeated clients/pages and only two later periods limit generalization. No iid tests or bootstrap confidence intervals.',
                   'Diagnostic comparisons are exploratory, not a new blind confirmation or a change of the primary prediction task.',
                   'Counterfactual missing labels may have additional unmodeled constraints; reported bounds allow any binary labels for exclusive unknowns.',
                   'No warehouse query or June access.']}


def paired_bounds07(frame,score_a,score_b,k=50):
    a,b=[np.asarray(s,dtype=float) for s in (score_a,score_b)]
    if any(s.ndim!=1 or len(s)!=len(frame) or not np.isfinite(s).all() for s in (a,b)):
        raise ValueError('Scores must be finite, aligned one-dimensional arrays.')
    if k<1:raise ValueError('Review budget must be positive.')
    if len(frame)<k:return {'status':'insufficient_candidates','budget':k,'n_candidates':len(frame)}
    if not frame.decoupling_signature.dropna().isin([0,1]).all():raise ValueError('Labels must be binary or unknown.')
    qa=np.zeros(len(frame),dtype=bool);qb=qa.copy()
    qa[np.argsort(-a,kind='stable')[:k]]=True
    qb[np.argsort(-b,kind='stable')[:k]]=True
    known=frame.decoupling_signature.notna().to_numpy()
    positive=frame.decoupling_signature.eq(1).fillna(False).to_numpy(dtype=bool)
    only_a=qa&~qb;only_b=qb&~qa;shared=qa&qb
    ha=int((only_a&positive).sum());hb=int((only_b&positive).sum())
    ua=int((only_a&~known).sum());ub=int((only_b&~known).sum())
    d=ha-hb
    return {'status':'evaluated','budget':k,'n_candidates':len(frame),
        'overlap_pages':int(shared.sum()),'jaccard':float(shared.sum()/(qa|qb).sum()),
        'shared_observed_positive_count':int((shared&positive).sum()),
        'shared_unknown_count':int((shared&~known).sum()),
        'a_only_pages':int(only_a.sum()),'b_only_pages':int(only_b.sum()),
        'a_only_observed_positives':ha,'b_only_observed_positives':hb,
        'a_only_unknown_count':ua,'b_only_unknown_count':ub,
        'a_observed_hits':int((qa&positive).sum()),'b_observed_hits':int((qb&positive).sum()),
        'a_unknown_count':int((qa&~known).sum()),'b_unknown_count':int((qb&~known).sum()),
        'known_contribution_difference':d/k,
        'paired_precision_difference_lower':(d-ub)/k,
        'paired_precision_difference_upper':(d+ua)/k,
        'point_difference_if_exclusive_outcomes_observed':d/k if ua+ub==0 else None}


def replay_scores07(train,test):
    # Exactly the four-feature LR construction used by score_models05.
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000,random_state=42))
    with warnings.catch_warnings(),threadpool_limits(limits=1):
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(train[FEATURES07].to_numpy(dtype=float),train.decoupling_signature.to_numpy(dtype=int))
        learned=model.predict_proba(test[FEATURES07].to_numpy(dtype=float))[:,1]
    rules,_=score_transparent_baselines(fit_pooled_reference(train),test)
    return {'Logistic Regression':learned,'Prior CTR':rules['Prior CTR'],
            'Visible pooled-CTR excess':rules['Visible pooled-CTR excess']}


def client_deletion07(frame,scores_a,scores_b,aliases,k=50):
    rows=[]
    for client in sorted(frame.client_hash_id.unique()):
        keep=~frame.client_hash_id.eq(client).to_numpy()
        rows.append({'deleted_client':aliases[str(client)],'n_removed_pages':int((~keep).sum()),
                     **paired_bounds07(frame.loc[keep],np.asarray(scores_a)[keep],np.asarray(scores_b)[keep],k)})
    return rows


def run_paired_audit07(episodes,validation):
    aliases={client:f'client_{i+1:03d}' for i,client in enumerate(sorted(
        {str(client) for frame in episodes.values() for client in frame.client_hash_id}))}
    origins=[]
    for fold in validation['folds']:
        arm=next(a for a in fold['arms'] if a['arm']=='known_clients')
        if arm['status']!='scored':raise ValueError('Expected the reviewed scored known-client arm.')
        lookup={str(t):frame for t,frame in episodes.items()}
        history=[lookup[t].sort_values(KEYS07,kind='stable').reset_index(drop=True) for t in fold['training_origins']]
        if len(history)!=3:raise ValueError('Three original training origins are required.')
        train=pd.concat(history,ignore_index=True)
        train=train.loc[train.decoupling_signature.notna()].copy()
        future=next(frame for t,frame in episodes.items() if str(t)==fold['origin'])
        test=future.loc[future.client_hash_id.isin(set(train.client_hash_id))].sort_values(KEYS07,kind='stable').reset_index(drop=True)
        digest=hashlib.sha256(pd.util.hash_pandas_object(test[KEYS07+FEATURES07+['decoupling_signature']],index=False).values.tobytes()).hexdigest()
        if digest!=arm['cohort_sha256']:raise ValueError('Fix05 known-client cohort/features/labels changed.')
        scores=replay_scores07(train,test)
        hashes={name:hashlib.sha256(np.asarray(score,dtype=float).tobytes()).hexdigest() for name,score in scores.items()}
        if any(value!=arm['prediction_sha256'][name] for name,value in hashes.items()):
            raise ValueError('Replayed predictions do not exactly match Fix05. Stop; do not reinterpret a changed ranking.')
        pairs=[]
        for a,b in PAIRS07:
            base=paired_bounds07(test,scores[a],scores[b])
            for name,prefix in [(a,'a'),(b,'b')]:
                original=next(row for row in arm['metrics'] if row['method']==name and row['minimum_prior_clicks']==1)
                q=next(q for q in original['queues'] if q['budget']==50)
                if (base[prefix+'_observed_hits'],base[prefix+'_unknown_count'])!=(q['observed_hits'],q['unknown_outcomes']):
                    raise ValueError('Original queue hit/unknown accounting differs after replay.')
            deletion=client_deletion07(test,scores[a],scores[b],aliases)
            valid=[row for row in deletion if row['status']=='evaluated']
            pairs.append({'method_a':a,'method_b':b,'full_portfolio':base,
                'client_deletions':deletion,
                'influence_summary':{'n_clients':len(deletion),'n_supported_deletions':len(valid),
                    'minimum_lower_bound_after_deletion':min((x['paired_precision_difference_lower'] for x in valid),default=None),
                    'maximum_upper_bound_after_deletion':max((x['paired_precision_difference_upper'] for x in valid),default=None),
                    'n_deletions_with_lower_bound_above_zero':sum(x['paired_precision_difference_lower']>0 for x in valid),
                    'n_deletions_with_upper_bound_below_zero':sum(x['paired_precision_difference_upper']<0 for x in valid)}})
        origins.append({'origin':fold['origin'],'arm':'known_clients','source_cohort_sha256':digest,
            'verified_prediction_sha256':hashes,'evaluation':arm['evaluation'],'pairs':pairs})
    return {'protocol':PROTOCOL07,'protocol_sha256':hashlib.sha256(json.dumps(PROTOCOL07,sort_keys=True).encode()).hexdigest(),
            'source_fix05_protocol_sha256':validation['protocol_sha256'],'origins':origins}
