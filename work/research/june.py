"""Additional retrospective June evaluation; definitions do not access data."""
from datetime import date,datetime,timezone
from pathlib import Path
import hashlib,json,os,re,uuid
import numpy as np
import pandas as pd
import duckdb
from .evaluation import (KEYS05,FEATURES05,METHODS05,build_episode05,train_origins05,
    heldout05,score_models05,metrics05,cohort_summary05,protocol_hash05)
from .warehouse import (parquet05m,sql_literal05m,file_hash05m,memory_settings05m,
    month_sql05m,validate_aggregate05m,checked_checkpoint05m,write_checkpoint05m)
from .pipeline import (verify_versions,verified_cache,PINS,SOURCE_REVISION,LEGACY_CACHE,
    digest_json,write_json)
from .paired import paired_bounds07,client_deletion07,PAIRS07
from .count_noise import summarize_count_noise06,COUNT_PROTOCOL06

JUNE_ORIGIN=date(2026,6,1)
JUNE_TRAINING=[date(2026,m,1) for m in (3,4,5)]
JUNE_PROTOCOL_SHA='cfbee182ac361efe2b932d9131f18c62dcb2694e3038a2ff4234df018ecf4c07'
JUNE_ARMS=['known_clients','naturally_unseen_clients','hash_heldout_clients']


def june_windows():
    return pd.DataFrame([{'origin':'2026-06-01','role':'outcome',
        'start_date':date(2026,6,1),'end_date':date(2026,6,28)}])


def june_source_paths():
    return ['hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/month=2026-06/*.parquet']


def june_feature_from_may(aggregate):
    required=set(KEYS05+['origin','role','raw_rows','raw_days','available_days',
        'bad_counts','bad_position','impressions','clicks','sum_position'])
    if not required.issubset(aggregate.columns):
        raise ValueError('June features require the unfiltered raw May window aggregate, not a May episode frame.')
    if set(aggregate.origin.astype(str))!={'2026-05-01'} or set(aggregate.role)!={'outcome'}:
        raise ValueError('Only the unfiltered May 1-28 outcome window can become June features.')
    result=aggregate.copy()
    result['origin']='2026-06-01';result['role']='feature'
    return result


def frame_sha_june(frame,labels=False):
    columns=KEYS05+FEATURES05+(['decoupling_signature'] if labels else [])
    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns],index=False).values.tobytes()).hexdigest()


def scores_sha_june(scores):
    return {name:hashlib.sha256(np.asarray(scores[name],dtype=float).tobytes()).hexdigest() for name in METHODS05}


def training_and_candidates_june(episodes,asof):
    if train_origins05(JUNE_ORIGIN)!=JUNE_TRAINING:
        raise RuntimeError('June maturity schedule changed.')
    if any(t not in episodes for t in JUNE_TRAINING):
        raise RuntimeError('Exactly the March, April and May episodes are required.')
    train=pd.concat([episodes[t] for t in JUNE_TRAINING],ignore_index=True)
    train=train.loc[train.decoupling_signature.notna()].copy()
    if not asof.decoupling_signature.isna().all():
        raise ValueError('Predictions must be frozen before June outcomes are attached.')
    result=[]
    for arm in JUNE_ARMS:
        tr=train.copy()
        if arm=='hash_heldout_clients':
            tr=tr.loc[~tr.client_hash_id.map(heldout05)]
            test=asof.loc[asof.client_hash_id.map(heldout05)].copy()
            if set(tr.client_hash_id)&set(test.client_hash_id):raise AssertionError('Hash client leakage.')
        else:
            seen=asof.client_hash_id.isin(set(tr.client_hash_id))
            test=asof.loc[seen if arm=='known_clients' else ~seen].copy()
        test=test.sort_values(KEYS05,kind='stable').reset_index(drop=True)
        record={'arm':arm,'training':cohort_summary05(tr),'n_as_of_candidates':len(test),
            'candidate_feature_sha256':frame_sha_june(test),
            'training_frame_sha256':frame_sha_june(tr,labels=True)}
        counts=tr.decoupling_signature.value_counts()
        if set(tr.origin)!={str(t) for t in JUNE_TRAINING}:status='insufficient_training_history'
        elif len(tr)<500 or tr.client_hash_id.nunique()<2 or min(counts.get(0,0),counts.get(1,0))<20:
            status='insufficient_training_support'
        elif not len(test):status='no_eligible_test_pages'
        else:status='scored'
        record['status']=status
        result.append((record,tr,test))
    return result


def freeze_predictions_june(con,episodes,asof,path,identity,progress=print):
    """Private durable prediction data, public-safe checksum receipt; no June input."""
    plan=training_and_candidates_june(episodes,asof)
    base={**identity,'kind':'june_predictions_before_outcomes',
        'training_origins':[str(t) for t in JUNE_TRAINING],
        'candidate_feature_sha256':frame_sha_june(asof),
        'arm_inputs':[record for record,_,_ in plan], 'methods':METHODS05,
        'versions':PINS,'june_outcomes_accessed_before_prediction_freeze':False}
    path=Path(path)
    ready=checked_checkpoint05m(path,base)
    if ready:
        progress('Verified private June predictions reused; no model refitting.')
        packed=con.execute('SELECT * FROM '+parquet05m([path])).df()
        records=ready['arms']
    else:
        parts=[];records=[]
        for record,tr,test in plan:
            record=dict(record)
            if record['status']=='scored':
                progress('Fitting June '+record['arm']+' before accessing June outcomes.')
                scores,audit=score_models05(tr,test)
                record.update(reference_audit=audit,prediction_sha256=scores_sha_june(scores))
                part=test[KEYS05+FEATURES05].copy();part['arm']=record['arm']
                for method in METHODS05:part[method]=scores[method]
                parts.append(part)
            records.append(record)
        packed=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=KEYS05+FEATURES05+['arm']+METHODS05)
        con.register('_june_private_predictions',packed)
        ready=write_checkpoint05m(con,'SELECT * FROM _june_private_predictions',path,
            {**base,'arms':records,'frozen_utc':datetime.now(timezone.utc).isoformat()},validate=False)
    restored={}
    if packed.duplicated(['arm']+KEYS05).any():raise RuntimeError('Duplicate frozen candidate scores.')
    for record,_,test in plan:
        saved=next(r for r in records if r['arm']==record['arm'])
        if any(saved.get(k)!=v for k,v in record.items()):raise RuntimeError('Frozen arm differs from its historical input plan.')
        if record['status']!='scored':continue
        part=packed.loc[packed.arm.eq(record['arm'])].sort_values(KEYS05,kind='stable').reset_index(drop=True)
        for column in ['impressions_prior28','clicks_prior28']:part[column]=part[column].astype('Int64')
        scores={name:part[name].to_numpy(dtype=float) for name in METHODS05}
        if frame_sha_june(part)!=record['candidate_feature_sha256'] or scores_sha_june(scores)!=saved['prediction_sha256']:
            raise RuntimeError('Frozen candidate alignment or prediction bytes changed.')
        restored[record['arm']]=(test,scores)
    if set(packed.arm)-{r['arm'] for r in records if r['status']=='scored'}:
        raise RuntimeError('Unexpected arm in private predictions.')
    return records,restored,ready


def acquire_june_outcomes(con,root,identity,*,synthetic_source=None,prediction_frozen_utc=None,progress=print):
    """One June partition only; retries reuse a verified local checkpoint."""
    root=Path(root);path=root/'outcomes.parquet'
    expected={**identity,'kind':'june_outcome_window','source_month':'2026-06',
        'start_date':'2026-06-01','end_date':'2026-06-28'}
    ready=checked_checkpoint05m(path,expected)
    if ready:
        if 'first_access_attempt_utc' not in ready:
            raise RuntimeError('June checkpoint lacks its original access-order receipt.')
        if prediction_frozen_utc is not None and datetime.fromisoformat(prediction_frozen_utc)>datetime.fromisoformat(ready['first_access_attempt_utc']):
            raise RuntimeError('Predictions were frozen after an earlier June access attempt; do not reuse this outcome checkpoint.')
        progress('Verified June outcome checkpoint reused; no remote query.')
        validate_aggregate05m(con,path)
        return path,ready,False
    # Shared across code-version directories in this VM, so a changed wrapper
    # cannot silently create a new "first" access after June was already read.
    ledger_path=root.parent/'first_access_attempt.json'
    ledger_identity={'protocol_sha256':identity['protocol_sha256'],
        'dataset_revision':identity['dataset_revision'],
        'kind':'first_June_outcome_acquisition_attempt_across_implementations'}
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
        if any(ledger.get(k)!=v for k,v in ledger_identity.items()):
            raise RuntimeError('June first-access ledger identity differs; preserve it and investigate.')
    else:
        ledger={**ledger_identity,'first_access_attempt_utc':datetime.now(timezone.utc).isoformat()}
        write_json(ledger_path,ledger)
    if prediction_frozen_utc is not None and datetime.fromisoformat(prediction_frozen_utc)>datetime.fromisoformat(ledger['first_access_attempt_utc']):
        raise RuntimeError('Predictions were frozen after an earlier June access attempt; stop for review.')
    expected={**expected,'first_access_attempt_utc':ledger['first_access_attempt_utc']}
    synthetic=identity['dataset_revision']=='synthetic_fixture_only'
    if synthetic:
        if synthetic_source is None:raise RuntimeError('Synthetic June run requires a local daily fixture; remote access is disabled.')
        sources=[str(synthetic_source)];revision=lambda:'synthetic_fixture_only'
    else:
        if synthetic_source is not None:raise ValueError('A synthetic source cannot replace warehouse outcomes.')
        token=os.environ.get('HF_TOKEN')
        if not token:
            try:
                from google.colab import userdata
                token=userdata.get('HF_TOKEN')
            except Exception:token=None
        if not token or not re.fullmatch(r'hf_[A-Za-z0-9]+',token):
            raise RuntimeError('Enable the approved HF_TOKEN Colab Secret; never paste a token.')
        from huggingface_hub import HfApi
        api=HfApi(token=token)
        try:con.execute("CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '"+token+"')")
        except Exception:raise RuntimeError('Could not configure private warehouse access.') from None
        del token
        revision=lambda:api.dataset_info('FlyRank/internship-warehouse').sha
        sources=june_source_paths()
    if revision()!=identity['dataset_revision']:raise RuntimeError('June release differs from verified historical cache.')
    con.register('_fix05_windows',june_windows())
    progress('Aggregating only June 1-28 from the June partition; predictions are already frozen.')
    stage=root/'pending'/('outcomes-'+uuid.uuid4().hex+'.parquet')
    metadata=write_checkpoint05m(con,month_sql05m(sources,'2026-06'),stage,expected)
    if revision()!=identity['dataset_revision']:
        raise RuntimeError('Warehouse revision changed during June aggregation; pending checkpoint is not approved.')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.replace(path.with_name('outcomes.unverified-'+uuid.uuid4().hex+'.parquet'))
    stage.replace(path);stage.with_suffix('.json').replace(path.with_suffix('.json'))
    return path,metadata,not synthetic


def report_june(episodes,feature,outcome,records,frozen,protocol):
    evaluated,audit=build_episode05(pd.concat([feature,outcome],ignore_index=True),JUNE_ORIGIN)
    asof,_=build_episode05(feature,JUNE_ORIGIN)
    if frame_sha_june(evaluated)!=frame_sha_june(asof):raise RuntimeError('Outcome attachment changed June candidates or features.')
    aliases={str(client):f'client_{i+1:03d}' for i,client in enumerate(sorted(evaluated.client_hash_id.unique()))}
    arms=[];paired_rows=[];noise_rows=[]
    for saved in records:
        row={**saved,'metrics':[]};arms.append(row)
        if saved['status']!='scored':continue
        before,scores=frozen[saved['arm']]
        # Explicit one-to-one key alignment, preserving the frozen ordering.
        test=before[KEYS05].merge(evaluated,on=KEYS05,how='left',validate='one_to_one',sort=False)
        if frame_sha_june(test)!=saved['candidate_feature_sha256']:
            raise RuntimeError('June labels are not aligned to frozen predictions.')
        row['evaluation']=cohort_summary05(test)
        row['cohort_sha256']=frame_sha_june(test,labels=True)
        for minimum in (1,5,20):
            mask=test.clicks_prior28.ge(minimum).to_numpy(dtype=bool)
            for name,score in scores.items():
                row['metrics'].append({'method':name,'minimum_prior_clicks':minimum,
                    'support_population':cohort_summary05(test.loc[mask]),**metrics05(test.loc[mask],score[mask])})
        details=[]
        for name,score in scores.items():
            selected=test.iloc[np.argsort(-score,kind='stable')[:50]] if len(test)>=50 else test.iloc[:0]
            for client,part in test.groupby('client_hash_id',sort=True):
                queue=selected.loc[selected.client_hash_id.eq(client)]
                details.append({'anonymous_client':aliases[str(client)],'method':name,**cohort_summary05(part),
                    'global_queue_pages':len(queue),'global_queue_observed_hits':int(queue.decoupling_signature.sum()),
                    'global_queue_unknown':int(queue.decoupling_signature.isna().sum())})
        row['anonymous_client_details']=details
        pairs=[]
        for a,b in PAIRS07:
            deletion=client_deletion07(test,scores[a],scores[b],aliases)
            valid=[x for x in deletion if x['status']=='evaluated']
            pairs.append({'method_a':a,'method_b':b,
                'role':'primary' if saved['arm']=='known_clients' and b=='Prior CTR' else 'secondary_descriptive',
                'full_portfolio':paired_bounds07(test,scores[a],scores[b]),'client_deletions':deletion,
                'influence_summary':{'n_clients':len(deletion),'n_supported_deletions':len(valid),
                    'minimum_lower_bound_after_deletion':min((x['paired_precision_difference_lower'] for x in valid),default=None),
                    'maximum_upper_bound_after_deletion':max((x['paired_precision_difference_upper'] for x in valid),default=None)}})
        paired_rows.append({'origin':'2026-06-01','arm':saved['arm'],'pairs':pairs,
            'source_cohort_sha256':row['cohort_sha256']})
        noise_rows.append({'origin':'2026-06-01','arm':saved['arm'],**summarize_count_noise06(test)})
    validation={'protocol':protocol,'protocol_sha256':JUNE_PROTOCOL_SHA,'cohort_audits':[audit],
        'folds':[{'origin':'2026-06-01','cutoff':'2026-05-28','training_origins':[str(t) for t in JUNE_TRAINING],
            'status':'completed_with_explicit_arm_statuses','arms':arms}],
        'limitations':protocol['limits']}
    paired={'source_protocol_sha256':JUNE_PROTOCOL_SHA,'direction':'LR precision@50 minus comparator precision@50',
        'interpretation':'Fixed-queue logical bounds, not confidence intervals. Client deletion uses fixed scores without refitting.',
        'anonymous_alias_scope':'This June run only; do not infer cross-run alias identity.','rows':paired_rows}
    count_protocol={**COUNT_PROTOCOL06,'version':'june-count-reference-v1',
        'status':'secondary_descriptive_diagnostic_after_frozen_June_prediction',
        'source':'Historical training plus June feature/outcome aggregates under the June addendum; never used for prediction.',
        'population':'Observed members of each June fixed as-of arm, restricted to total clicks <=200.'}
    noise={'protocol':count_protocol,'protocol_sha256':digest_json(count_protocol),
        'source_protocol_sha256':JUNE_PROTOCOL_SHA,'rows':noise_rows}
    return validation,paired,noise


def run_june(protocol_receipt,extractor_sha,implementation_sha,*,cache_path=None,synthetic=False,
             synthetic_source=None,output_root=None,progress=print):
    verify_versions()
    protocol=protocol_receipt['protocol']
    if digest_json(protocol)!=JUNE_PROTOCOL_SHA or protocol_receipt['protocol_sha256']!=JUNE_PROTOCOL_SHA:
        raise RuntimeError('June protocol differs from the version committed before access.')
    for value in (extractor_sha,implementation_sha):
        if not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('Expected implementation SHA256.')
    root=Path(output_root or ('work/research/results/synthetic_june' if synthetic else 'work/research/results/june'))
    run=root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run.mkdir(parents=True,exist_ok=False)
    write_json(run/'protocol.json',protocol_receipt)
    write_json(run/'execution_plan.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_commit':'6cba97f','protocol_sha256':JUNE_PROTOCOL_SHA,'implementation_sha256':implementation_sha,
        'mode':'synthetic_fixture_only' if synthetic else 'user_colab_warehouse',
        'prior_June_user_exposure':'uncertain','June_outcome_access_in_this_run_before_execution_plan':False,
        'previous_workflow_access_status_at_plan_creation':'not_assessed; checked against the immutable access ledger later'})
    progress('Committed June protocol and execution identity saved before cache or source access.')
    private=Path('work/private_cache/study/june_synthetic' if synthetic else 'work/private_cache/study/june')/implementation_sha[:16]
    candidate=Path(cache_path) if cache_path else (LEGACY_CACHE if LEGACY_CACHE.exists() else
        Path('work/private_cache/study/development')/extractor_sha[:16]/'windows.parquet')
    if not candidate.exists():raise RuntimeError('Verified October-May cache is missing. Restore it or run development cache preparation first; no automatic historical rescan.')
    historical=verified_cache(candidate,extractor_sha,synthetic)
    identity={'protocol_sha256':JUNE_PROTOCOL_SHA,'implementation_sha256':implementation_sha,
        'historical_cache_sha256':historical['sha256'],'dataset_revision':historical['dataset_revision']}
    con=duckdb.connect()
    try:
        settings=memory_settings05m(con,private)
        validate_aggregate05m(con,candidate,merged=True)
        episodes={}
        for origin in JUNE_TRAINING:
            aggregate=con.execute('SELECT * FROM '+parquet05m([candidate])+' WHERE origin=?',[str(origin)]).df()
            episodes[origin],_=build_episode05(aggregate,origin)
        may=con.execute('SELECT * FROM '+parquet05m([candidate])+" WHERE origin='2026-05-01' AND role='outcome'").df()
        feature=june_feature_from_may(may);del may,aggregate
        asof,_=build_episode05(feature,JUNE_ORIGIN)
        records,frozen,prediction_receipt=freeze_predictions_june(con,episodes,asof,private/'predictions.parquet',identity,progress)
        write_json(run/'predictions_frozen_receipt.json',prediction_receipt)
        progress('All supported June rankings are frozen. No June outcomes used in fitting or candidate selection.')
        outcome_path,outcome_receipt,queried=acquire_june_outcomes(con,private,identity,
            synthetic_source=synthetic_source,prediction_frozen_utc=prediction_receipt['frozen_utc'],progress=progress)
        outcome=con.execute('SELECT * FROM '+parquet05m([outcome_path])).df()
        if len(outcome) and (set(outcome.origin.astype(str))!={'2026-06-01'} or set(outcome.role)!={'outcome'}):
            raise RuntimeError('Unexpected June outcome window roster.')
    finally:con.close()
    validation,paired,noise=report_june(episodes,feature,outcome,records,frozen,protocol)
    order_verified=datetime.fromisoformat(prediction_receipt['frozen_utc'])<=datetime.fromisoformat(outcome_receipt['first_access_attempt_utc'])
    if not order_verified:raise RuntimeError('Prediction freeze does not precede the first recorded June acquisition attempt.')
    execution={'mode':'synthetic_fixture_only' if synthetic else 'user_colab_warehouse',
        'implementation_sha256':implementation_sha,'historical_source':historical,'june_source':outcome_receipt,
        'june_partition_queried_this_run':queried,'predictions_frozen_before_June_outcome_access':order_verified,
        'prediction_frozen_utc':prediction_receipt['frozen_utc'],
        'first_June_access_attempt_utc':outcome_receipt['first_access_attempt_utc'],
        'access_order_scope':'This Colab VM and addendum across implementation checkpoints; no claim about exposure before this VM or a blind test.',
        'source_months_new':['2026-06'],'final_month_sample_used':False,'versions':PINS,
        'memory_settings':settings,'prior_user_exposure':'uncertain','independent_prospective_test':False,
        'private_identifiers_or_prediction_arrays_exported':False}
    validation['execution']=execution
    for name,value in [('june_validation.json',validation),('paired_queue_audit.json',paired),('count_noise_audit.json',noise)]:
        write_json(run/name,value)
    write_json(run/'run_manifest.json',{'status':'complete','execution':execution,
        'files':{p.name:file_hash05m(p) for p in sorted(run.glob('*.json'))},
        'scored_arms':sum(r['status']=='scored' for r in records)})
    progress('JUNE COMPLETE: aggregate-only results saved in '+str(run))
    return run
