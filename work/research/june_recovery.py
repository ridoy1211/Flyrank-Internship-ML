"""Execution amendment: validate June outcomes on the pre-frozen candidate universe."""
from datetime import datetime,timezone
from pathlib import Path
import json,os,re
import duckdb
import pandas as pd
from .evaluation import KEYS05,FEATURES05,build_episode05
from .warehouse import (parquet05m,file_hash05m,validate_aggregate05m,write_checkpoint05m,
    checked_checkpoint05m,memory_settings05m,month_sql05m)
from .pipeline import verified_cache,write_json,SOURCE_REVISION,PINS,LEGACY_CACHE,verify_versions,digest_json
from .june import (JUNE_ORIGIN,JUNE_TRAINING,JUNE_PROTOCOL_SHA,june_feature_from_may,
    frame_sha_june,freeze_predictions_june,report_june,june_windows,june_source_paths)

ORIGINAL_JUNE_IMPLEMENTATION='39b0eccfc593ef6970e19ec3519fd6dad70e03bf9fae96b092542d0a5db4d984'
QUALITY_CONDITION_JUNE="""client_hash_id IS NULL OR content_hash_id IS NULL OR raw_rows!=raw_days OR
    raw_days!=bit_count(row_mask) OR available_days!=bit_count(available_mask) OR raw_days>28 OR
    row_mask>=268435456 OR available_days>raw_days OR (row_mask|available_mask)!=row_mask OR bad_counts>0"""


def scope_cached_june_outcomes(con,raw_path,prediction_path,asof,destination,identity):
    """No repairs/deduplication: select pre-existing candidates, then enforce all guards."""
    raw_path=Path(raw_path);prediction_path=Path(prediction_path)
    global_sha=file_hash05m(raw_path)
    keys_features=','.join(KEYS05+FEATURES05)
    union=con.execute('SELECT DISTINCT '+keys_features+' FROM '+parquet05m([prediction_path])).df()
    if union.duplicated(KEYS05).any():raise RuntimeError('Conflicting frozen features for the same candidate key.')
    for column in ['impressions_prior28','clicks_prior28']:union[column]=union[column].astype('Int64')
    union=union.sort_values(KEYS05,kind='stable').reset_index(drop=True)
    if frame_sha_june(union)!=frame_sha_june(asof):
        raise RuntimeError('Frozen score-key union differs from the full May-derived as-of universe; stop recovery.')
    con.execute('CREATE OR REPLACE TEMP VIEW _june_failed_global AS SELECT * FROM '+parquet05m([raw_path]))
    con.execute('CREATE OR REPLACE TEMP VIEW _june_frozen_keys AS SELECT DISTINCT '+','.join(KEYS05)+' FROM '+parquet05m([prediction_path]))
    columns='COUNT(*) AS n_groups,COUNT(*) FILTER(WHERE '+QUALITY_CONDITION_JUNE+') AS n_failed_groups'
    total=con.execute('SELECT '+columns+',COUNT(*) FILTER(WHERE raw_rows!=raw_days) AS duplicate_daily_key_groups,SUM(raw_rows-raw_days) AS extra_daily_rows FROM _june_failed_global').df().to_dict('records')[0]
    selected=con.execute('SELECT '+columns+' FROM _june_failed_global JOIN _june_frozen_keys USING('+','.join(KEYS05)+')').df().to_dict('records')[0]
    if selected['n_failed_groups']:
        raise RuntimeError('A source-quality failure overlaps the frozen candidate population; recovery is forbidden.')
    roster=con.execute('SELECT DISTINCT origin,role FROM _june_failed_global').fetchall()
    if set(roster)!={('2026-06-01','outcome')}:raise RuntimeError('Unexpected outcome window in the failed source artifact.')
    receipt_identity={**identity,'kind':'June_outcomes_scoped_to_pre_frozen_candidates',
        'global_failed_aggregate_sha256':global_sha,'frozen_prediction_file_sha256':file_hash05m(prediction_path),
        'full_candidate_feature_sha256':frame_sha_june(asof),
        'source_month':'2026-06','start_date':'2026-06-01','end_date':'2026-06-28'}
    ready=checked_checkpoint05m(destination,receipt_identity)
    if ready is None:
        query='SELECT a.* FROM _june_failed_global a JOIN _june_frozen_keys k USING('+','.join(KEYS05)+')'
        ready=write_checkpoint05m(con,query,destination,{**receipt_identity,
            'global_source_quality':total,'candidate_source_quality':selected,
            'data_deduplication_performed':False,'candidate_pages_removed':0,
            'original_global_aggregate_preserved':True})
    validate_aggregate05m(con,destination)
    if file_hash05m(raw_path)!=global_sha:raise RuntimeError('Global failed aggregate changed during recovery.')
    return ready


def recover_june_fixed_candidates(protocol_receipt,extractor_sha,recovery_sha,*,history_cache=None,
        original_private_root=None,synthetic=False,revision_check=None,output_root=None,progress=print):
    verify_versions()
    if protocol_receipt['protocol_sha256']!=JUNE_PROTOCOL_SHA or digest_json(protocol_receipt['protocol'])!=JUNE_PROTOCOL_SHA:
        raise RuntimeError('Preserve the committed June protocol unchanged.')
    original=Path(original_private_root or 'work/private_cache/study/june/39b0eccfc593ef69')
    candidate=Path(history_cache) if history_cache else (LEGACY_CACHE if LEGACY_CACHE.exists() else
        Path('work/private_cache/study/development')/extractor_sha[:16]/'windows.parquet')
    historical=verified_cache(candidate,extractor_sha,synthetic)
    expected_revision='synthetic_fixture_only' if synthetic else SOURCE_REVISION
    if historical['dataset_revision']!=expected_revision:raise RuntimeError('Historical release mismatch.')
    identity={'protocol_sha256':JUNE_PROTOCOL_SHA,'implementation_sha256':ORIGINAL_JUNE_IMPLEMENTATION,
        'historical_cache_sha256':historical['sha256'],'dataset_revision':expected_revision}
    prediction_path=original/'predictions.parquet'
    if not prediction_path.exists() or not prediction_path.with_suffix('.json').exists():
        raise RuntimeError('Original frozen predictions are required; recovery must never refit models.')
    before=checked_checkpoint05m(prediction_path,identity)
    ledger=json.loads((original.parent/'first_access_attempt.json').read_text(encoding='utf-8'))
    if any(ledger.get(k)!=identity[k] for k in ['protocol_sha256','dataset_revision']):
        raise RuntimeError('Original access ledger differs from the frozen study.')
    if datetime.fromisoformat(before['frozen_utc'])>datetime.fromisoformat(ledger['first_access_attempt_utc']):
        raise RuntimeError('Original predictions postdate June acquisition; stop recovery.')
    raw=list((original/'pending').glob('*.parquet'))
    if len(raw)!=1:raise RuntimeError('Expected one preserved failed June aggregate; resolve ambiguity before recovery.')
    root=Path(output_root or ('work/research/results/synthetic_june_recovery' if synthetic else 'work/research/results/june_recovery'))
    run=root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run.mkdir(parents=True,exist_ok=False)
    amendment={'version':'June-candidate-source-scope-execution-amendment-v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'reason':'Global June source grain failed outside the frozen candidate universe.',
        'committed_protocol_sha256':JUNE_PROTOCOL_SHA,'original_implementation_sha256':ORIGINAL_JUNE_IMPLEMENTATION,
        'recovery_implementation_sha256':recovery_sha,
        'scientific_protocol_or_threshold_changes':False,'new_model_fit_or_tuning':False,
        'source_scope':'All and only the original May-derived candidate keys, before any outcome filtering.',
        'required_guards':['Exact candidate key and four-feature equality to original frozen union.',
            'Zero duplicate/count/coverage failures in candidate-scoped June outcomes.',
            'Preserve and hash failed global aggregate; no deduplication or raw-data requery.',
            'Left-join outcomes to every frozen candidate; retain absent/incomplete labels as unknown.'],
        'disclosure':'Execution-level source scoping after a global source-quality stop; not a blind test or a predeclared data-cleaning result.'}
    write_json(run/'execution_amendment.json',amendment)
    write_json(run/'protocol.json',protocol_receipt)
    if revision_check is None:
        if synthetic:raise ValueError('Synthetic recovery must provide a local revision callback.')
        token=os.environ.get('HF_TOKEN')
        if not token:
            from google.colab import userdata
            token=userdata.get('HF_TOKEN')
        if not token or not re.fullmatch(r'hf_[A-Za-z0-9]+',token):raise RuntimeError('Enable the approved Colab Secret.')
        from huggingface_hub import HfApi
        api=HfApi(token=token);del token
        revision_check=lambda:api.dataset_info('FlyRank/internship-warehouse').sha
    if revision_check()!=expected_revision:
        raise RuntimeError('June source revision changed after the original failed query; no mixed-release recovery.')
    con=duckdb.connect()
    try:
        settings=memory_settings05m(con,original/'scope_recovery')
        episodes={}
        for origin in JUNE_TRAINING:
            aggregate=con.execute('SELECT * FROM '+parquet05m([candidate])+' WHERE origin=?',[str(origin)]).df()
            episodes[origin],_=build_episode05(aggregate,origin)
        may=con.execute('SELECT * FROM '+parquet05m([candidate])+" WHERE origin='2026-05-01' AND role='outcome'").df()
        feature=june_feature_from_may(may)
        asof,_=build_episode05(feature,JUNE_ORIGIN)
        records,frozen,prediction_receipt=freeze_predictions_june(con,episodes,asof,prediction_path,identity,progress)
        scoped_path=original/'scope_recovery/candidate_outcomes.parquet'
        outcome_receipt=scope_cached_june_outcomes(con,raw[0],prediction_path,asof,scoped_path,
            {**identity,'execution_recovery_sha256':recovery_sha,
             'first_access_attempt_utc':ledger['first_access_attempt_utc']})
        outcome=con.execute('SELECT * FROM '+parquet05m([scoped_path])).df()
    finally:con.close()
    if revision_check()!=expected_revision:raise RuntimeError('Source revision changed during scoped recovery.')
    validation,paired,noise=report_june(episodes,feature,outcome,records,frozen,protocol_receipt['protocol'])
    execution={'mode':'synthetic_fixture_only' if synthetic else 'user_colab_warehouse',
        'implementation_sha256':ORIGINAL_JUNE_IMPLEMENTATION,'execution_recovery_sha256':recovery_sha,
        'historical_source':historical,'june_source':outcome_receipt,
        'june_partition_queried_this_run':False,'new_raw_warehouse_query_during_recovery':False,
        'model_fits_during_recovery':0,'source_revision_verified_after_original_query':True,
        'predictions_frozen_before_June_outcome_access':True,
        'prediction_frozen_utc':prediction_receipt['frozen_utc'],
        'first_June_access_attempt_utc':ledger['first_access_attempt_utc'],
        'source_months_new':['2026-06'],'final_month_sample_used':False,'versions':PINS,
        'memory_settings':settings,'prior_user_exposure':'uncertain','independent_prospective_test':False,
        'private_identifiers_or_prediction_arrays_exported':False,
        'source_scoping_execution_amendment':amendment['version'],
        'global_source_quality_failure_disclosed':True,'all_candidate_source_quality_guards_passed':True}
    validation['execution']=execution
    for name,value in [('june_validation.json',validation),('paired_queue_audit.json',paired),('count_noise_audit.json',noise),
        ('predictions_frozen_receipt.json',prediction_receipt)]:write_json(run/name,value)
    write_json(run/'run_manifest.json',{'status':'complete_with_disclosed_source_scoping_recovery',
        'execution':execution,'files':{p.name:file_hash05m(p) for p in sorted(run.glob('*.json'))},
        'scored_arms':sum(r['status']=='scored' for r in records)})
    progress('JUNE SCOPED RECOVERY COMPLETE: original predictions and all frozen candidates retained.')
    return run


def rerun_june_after_runtime_reset(protocol_receipt,extractor_sha,recovery_sha,*,
        history_cache=None,synthetic=False,synthetic_source=None,output_root=None,
        private_root=None,progress=print):
    """Repeat the fixed computation; never represent lost checkpoints as recovered."""
    verify_versions()
    if protocol_receipt['protocol_sha256']!=JUNE_PROTOCOL_SHA or digest_json(protocol_receipt['protocol'])!=JUNE_PROTOCOL_SHA:
        raise RuntimeError('The committed scientific protocol must remain unchanged.')
    if not re.fullmatch('[a-f0-9]{64}',recovery_sha):raise ValueError('Expected recovery source SHA256.')
    expected_revision='synthetic_fixture_only' if synthetic else SOURCE_REVISION
    root=Path(output_root or ('work/research/results/synthetic_june_recovery' if synthetic else 'work/research/results/june_recovery'))
    run=root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run.mkdir(parents=True,exist_ok=False)
    disclosure={'version':'June-runtime-reset-rerun-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'reason':'The previous Colab runtime expired; its private predictions and source aggregate were lost.',
        'previous_June_source_inspection':{'duplicate_daily_key_groups':355,'extra_daily_rows':5680,
            'failed_groups_in_frozen_candidate_population':0,'ranking_performance_inspected':False},
        'original_implementation_sha256':ORIGINAL_JUNE_IMPLEMENTATION,
        'recovery_implementation_sha256':recovery_sha,'protocol_sha256':JUNE_PROTOCOL_SHA,
        'scientific_protocol_or_threshold_changes':False,'new_model_search_or_tuning':False,
        'source_scope':'All and only May-derived candidate keys, determined before outcome attachment.',
        'data_deduplication_performed':False,
        'interpretation':'Retrospective rerun after known June source-quality inspection; original prediction bytes/timestamps cannot be recovered or verified.'}
    write_json(run/'execution_amendment.json',disclosure);write_json(run/'protocol.json',protocol_receipt)
    progress('RUNTIME-RESET RERUN disclosed before execution. The original private checkpoints are unavailable.')
    candidate=Path(history_cache) if history_cache else Path('work/private_cache/study/development')/extractor_sha[:16]/'windows.parquet'
    historical=verified_cache(candidate,extractor_sha,synthetic)
    private=Path(private_root or ('work/private_cache/study/june_reset_synthetic' if synthetic else 'work/private_cache/study/june_reset_rerun'))/recovery_sha[:16]
    identity={'protocol_sha256':JUNE_PROTOCOL_SHA,'implementation_sha256':ORIGINAL_JUNE_IMPLEMENTATION,
        'historical_cache_sha256':historical['sha256'],'dataset_revision':expected_revision,
        'execution_recovery_sha256':recovery_sha,'runtime_reset_rerun':True}
    con=duckdb.connect()
    try:
        settings=memory_settings05m(con,private)
        validate_aggregate05m(con,candidate,merged=True)
        episodes={}
        for origin in JUNE_TRAINING:
            aggregate=con.execute('SELECT * FROM '+parquet05m([candidate])+' WHERE origin=?',[str(origin)]).df()
            episodes[origin],_=build_episode05(aggregate,origin)
        may=con.execute('SELECT * FROM '+parquet05m([candidate])+" WHERE origin='2026-05-01' AND role='outcome'").df()
        feature=june_feature_from_may(may);asof,_=build_episode05(feature,JUNE_ORIGIN)
        prediction_path=private/'predictions.parquet'
        reused_predictions=checked_checkpoint05m(prediction_path,identity) is not None
        records,frozen,prediction_receipt=freeze_predictions_june(con,episodes,asof,prediction_path,identity,progress)
        write_json(run/'predictions_frozen_receipt.json',prediction_receipt)
        raw_path=private/'global_june_aggregate.parquet'
        raw_identity={**identity,'kind':'global_June_source_preserved_before_candidate_quality_validation',
            'source_month':'2026-06','start_date':'2026-06-01','end_date':'2026-06-28'}
        raw_receipt=checked_checkpoint05m(raw_path,raw_identity)
        queried=raw_receipt is None
        if queried:
            if synthetic:
                if synthetic_source is None:raise ValueError('Local synthetic daily source required.')
                sources=[str(synthetic_source)];revision=lambda:expected_revision
            else:
                if synthetic_source is not None:raise ValueError('No source substitution in warehouse mode.')
                token=os.environ.get('HF_TOKEN')
                if not token:
                    from google.colab import userdata
                    token=userdata.get('HF_TOKEN')
                if not token or not re.fullmatch(r'hf_[A-Za-z0-9]+',token):raise RuntimeError('Enable the existing HF_TOKEN Colab Secret.')
                from huggingface_hub import HfApi
                api=HfApi(token=token)
                try:con.execute("CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '"+token+"')")
                except Exception:raise RuntimeError('Could not configure warehouse access.') from None
                del token
                revision=lambda:api.dataset_info('FlyRank/internship-warehouse').sha
                sources=june_source_paths()
            if revision()!=expected_revision:raise RuntimeError('Source release differs from the fixed historical release.')
            started=datetime.now(timezone.utc).isoformat()
            con.register('_fix05_windows',june_windows())
            progress('Fixed rankings saved. Aggregating June 1-28 once; preserving the global quality audit.')
            raw_receipt=write_checkpoint05m(con,month_sql05m(sources,'2026-06'),raw_path,
                {**raw_identity,'acquisition_started_utc':started},validate=False)
            if revision()!=expected_revision:raise RuntimeError('Source release changed during acquisition; stop.')
            raw_receipt['source_revision_verified_after_query']=True
            write_json(raw_path.with_suffix('.json'),raw_receipt)
        if raw_receipt.get('source_revision_verified_after_query') is not True:
            raise RuntimeError('Global source checkpoint lacks a completed source-revision verification.')
        if datetime.fromisoformat(prediction_receipt['frozen_utc'])>datetime.fromisoformat(raw_receipt['acquisition_started_utc']):
            raise RuntimeError('Rerun predictions postdate the local outcome acquisition.')
        outcome_receipt=scope_cached_june_outcomes(con,raw_path,prediction_path,asof,
            private/'candidate_outcomes.parquet',identity)
        outcome=con.execute('SELECT * FROM '+parquet05m([private/'candidate_outcomes.parquet'])).df()
    finally:con.close()
    validation,paired,noise=report_june(episodes,feature,outcome,records,frozen,protocol_receipt['protocol'])
    execution={'mode':'synthetic_fixture_only' if synthetic else 'user_colab_warehouse',
        'implementation_sha256':ORIGINAL_JUNE_IMPLEMENTATION,'execution_recovery_sha256':recovery_sha,
        'historical_source':historical,'june_source':outcome_receipt,'global_june_source':raw_receipt,
        'june_partition_queried_this_run':bool(queried and not synthetic),
        'model_fits_this_run':0 if reused_predictions else sum(r['status']=='scored' for r in records),
        'original_prediction_checkpoint_recovered':False,'runtime_reset_rerun':True,
        'previous_June_source_inspection_disclosed':True,'independent_prospective_test':False,
        'prediction_frozen_utc':prediction_receipt['frozen_utc'],
        'this_rerun_outcome_acquisition_started_utc':raw_receipt['acquisition_started_utc'],
        'access_order_scope':'Only this rerun; June source defects were already known from the expired runtime.',
        'final_month_sample_used':False,'versions':PINS,'memory_settings':settings,
        'private_identifiers_or_prediction_arrays_exported':False,
        'all_candidate_source_quality_guards_passed':True}
    validation['execution']=execution
    for name,value in [('june_validation.json',validation),('paired_queue_audit.json',paired),('count_noise_audit.json',noise)]:
        write_json(run/name,value)
    write_json(run/'run_manifest.json',{'status':'complete_retrospective_rerun_with_disclosed_source_scoping',
        'execution':execution,'files':{p.name:file_hash05m(p) for p in sorted(run.glob('*.json'))},
        'scored_arms':sum(r['status']=='scored' for r in records)})
    progress('JUNE RERUN COMPLETE. Candidate quality checks passed; all fixed candidates retained.')
    return run
