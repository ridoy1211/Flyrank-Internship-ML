"""Development analysis entry point. Public exports contain aggregate results only."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, importlib, importlib.metadata, json, os, platform, re
import duckdb
from .evaluation import (PROTOCOL05, EPISODES05, MONTHS05, protocol_hash05,
    windows05, build_episode05, run_chronological05)
from .warehouse import (file_hash05m, checked_checkpoint05m, memory_settings05m,
    build_cache05m, parquet05m, validate_aggregate05m)
from .count_noise import COUNT_PROTOCOL06, audit_fixed_episodes06
from .paired import PROTOCOL07, run_paired_audit07

PINS = {'numpy':'2.2.6','pandas':'2.2.3','scipy':'1.16.3',
    'scikit-learn':'1.7.2','matplotlib':'3.10.7','duckdb':'1.4.1','huggingface_hub':'1.0.1'}
SOURCE_REVISION = '50cbf7c3909d07be4d1b5906b4d09e882e5acbf2'
LEGACY_EXTRACTOR = '7b1c93304051641d59bd35b7fcb8bfb5dc85c54571b36c4736eb41453d1e71cd'
LEGACY_CACHE = Path('work/private_cache/fix05/memory_recovery_v1/windows.parquet')
LEGACY_CACHE_SHA = 'f85b104bc9281ee54903c025a5c835699dc98a8c5afd02f72e669594f09d2d4c'


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def verify_versions():
    errors=[]
    for name,expected in PINS.items():
        module=importlib.import_module('sklearn' if name=='scikit-learn' else name)
        loaded=getattr(module,'__version__',None)
        installed=importlib.metadata.version(name)
        if loaded!=expected or installed!=expected:
            errors.append(f'{name}: loaded={loaded}, installed={installed}, expected={expected}')
    if errors:
        raise RuntimeError('Restart the Colab session after setup, then Run all. '+ '; '.join(errors))


def development_protocol():
    return {'version':'clean-development-v1',
        'status':'consolidated_retrospective_analysis_not_preregistration',
        'forecast_protocol':PROTOCOL05,'forecast_protocol_sha256':protocol_hash05(),
        'count_reference_protocol':COUNT_PROTOCOL06,'paired_diagnostic_protocol':PROTOCOL07,
        'june_read':False,'june_prior_user_exposure':'uncertain',
        'purpose':'Reproduce repaired February-May analyses without continuation-cell dependencies.',
        'interpretation':'No blind-test, causal, equivalence or population-wide ML-failure claim.'}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verified_cache(path,extractor_sha,synthetic=False):
    """Validate provenance and bytes before loading any page-key aggregates."""
    path=Path(path)
    receipt_path=path.with_suffix('.json')
    if not receipt_path.exists():raise RuntimeError('Cache receipt is missing; preserve files and investigate.')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    producers={extractor_sha} if synthetic else {extractor_sha,LEGACY_EXTRACTOR}
    if receipt.get('implementation_sha256') not in producers:
        raise RuntimeError('Unrecognized cache producer; do not mix extraction implementations.')
    expected={'protocol_sha256':protocol_hash05(),'source_months':MONTHS05,
        'kind':'combined_verified_windows',
        'dataset_revision':'synthetic_fixture_only' if synthetic else SOURCE_REVISION}
    if any(receipt.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('Cache source, date scope or protocol differs from the fixed development study.')
    checked_checkpoint05m(path,{**expected,'implementation_sha256':receipt['implementation_sha256']})
    if not synthetic and receipt['implementation_sha256']==LEGACY_EXTRACTOR and receipt['sha256']!=LEGACY_CACHE_SHA:
        raise RuntimeError('Legacy cache bytes differ from the reviewed warehouse run.')
    return receipt


def acquire_development_cache(con,extractor_sha,synthetic=False,cache_path=None,progress=print):
    root=Path('work/private_cache/study/development')/extractor_sha[:16]
    candidate=Path(cache_path) if cache_path is not None else (LEGACY_CACHE if LEGACY_CACHE.exists() else root/'windows.parquet')
    if candidate.exists():
        receipt=verified_cache(candidate,extractor_sha,synthetic)
        progress('Verified local cache reused; no warehouse access.')
        return candidate,receipt
    if synthetic:raise RuntimeError('Synthetic mode requires a prepared private fixture cache; remote access is disabled.')
    if cache_path is not None:raise RuntimeError('Explicit cache path does not exist; no remote fallback.')
    api=None
    def revision():
        nonlocal api
        if api is None:
            token=os.environ.get('HF_TOKEN')
            if not token:
                try:
                    from google.colab import userdata
                    token=userdata.get('HF_TOKEN')
                except Exception:token=None
            if not token or not re.fullmatch(r'hf_[A-Za-z0-9]+',token):
                raise RuntimeError('Enable the HF_TOKEN Colab Secret; do not paste a token.')
            from huggingface_hub import HfApi
            api=HfApi(token=token)
            try:con.execute("CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '"+token+"')")
            except Exception:raise RuntimeError('Could not configure private warehouse access.') from None
        value=api.dataset_info('FlyRank/internship-warehouse').sha
        if value!=SOURCE_REVISION:raise RuntimeError('Warehouse release changed; do not combine releases.')
        return value
    def sources(month):
        if month not in MONTHS05:raise ValueError('Month outside the fixed development protocol.')
        return ['hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/month='+month+'/*.parquet']
    return build_cache05m(con,windows05(),MONTHS05,root,protocol_hash05(),extractor_sha,sources,revision,progress)


def run_development(extractor_sha,implementation_sha,*,cache_path=None,synthetic=False,
                    output_root=None,progress=print):
    """Freeze receipt, load/build verified cache, score, diagnose, export aggregates."""
    verify_versions()
    for value in (extractor_sha,implementation_sha):
        if not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('Expected SHA256 implementation identity.')
    if protocol_hash05()!='fa08ab634857e99ecdd9b194c93105795258b31ddaed49454c55f2d8a478e0a8':
        raise RuntimeError('Forecast protocol changed before data access.')
    protocol=development_protocol()
    root=Path(output_root or ('work/research/results/synthetic' if synthetic else 'work/research/results/development'))
    run=root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/'protocol.json',{'protocol':protocol,'protocol_sha256':digest_json(protocol),
        'created_utc':datetime.now(timezone.utc).isoformat(),'implementation_sha256':implementation_sha})
    progress('Development protocol saved before cache access. June remains excluded.')
    con=duckdb.connect()
    try:
        settings=memory_settings05m(con,Path('work/private_cache/study/spill'))
        cache,receipt=acquire_development_cache(con,extractor_sha,synthetic,cache_path,progress)
        validate_aggregate05m(con,cache,merged=True)
        windows=con.execute('SELECT DISTINCT origin,role FROM '+parquet05m([cache])).fetchall()
        expected={(str(t),r) for t in EPISODES05 for r in ('feature','outcome')}
        if set(windows)!=expected:raise RuntimeError('Cached window roster differs from the fixed fourteen windows.')
        episodes={};audits=[]
        for origin in EPISODES05:
            aggregate=con.execute('SELECT * FROM '+parquet05m([cache])+' WHERE origin=?',[str(origin)]).df()
            episodes[origin],audit=build_episode05(aggregate,origin);audits.append(audit)
            del aggregate
    finally:con.close()
    progress('Private aggregate checks passed. Fitting fixed historical models.')
    validation=run_chronological05(episodes,audits,progress=progress)
    execution={'mode':'synthetic_fixture_only' if synthetic else 'user_colab_warehouse',
        'implementation_sha256':implementation_sha,'extractor_sha256':extractor_sha,
        'source':receipt,'versions':PINS,'python':platform.python_version(),
        'memory_settings':settings,'june_read':False,'raw_identifiers_or_scores_exported':False}
    validation['execution']=execution
    write_json(run/'chronological_validation.json',validation)
    progress('Auditing conditional count allocation on unchanged observed cohorts.')
    noise=audit_fixed_episodes06(episodes,validation)
    write_json(run/'count_noise_audit.json',noise)
    progress('Verifying paired LR/rule prediction hashes and client deletions.')
    paired=run_paired_audit07(episodes,validation)
    write_json(run/'paired_queue_audit.json',paired)
    manifest={'status':'complete','protocol_sha256':digest_json(protocol),'execution':execution,
        'files':{p.name:file_hash05m(p) for p in sorted(run.glob('*.json'))},
        'scored_arms':sum(a['status']=='scored' for f in validation['folds'] for a in f['arms']),
        'count_reference_rows':len(noise['rows']),
        'paired_comparisons':sum(len(o['pairs']) for o in paired['origins'])}
    write_json(run/'run_manifest.json',manifest)
    progress('COMPLETE: aggregate results saved in '+str(run))
    return run
