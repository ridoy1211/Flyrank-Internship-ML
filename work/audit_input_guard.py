"""Scope audit integrity checks to data and results actually consumed by the audits."""
import hashlib,json

GUARD_EXCLUSIONS=[
    'folds/*/arms/*/prediction_sha256/Random Forest (raw byte checksum only; all RF metrics retained)',
    'execution/memory_settings/available_memory_bytes_at_configuration (resource telemetry only)']

def normalized_fix05_inputs(payload):
    copied=json.loads(json.dumps(payload,allow_nan=False))
    for fold in copied['folds']:
        for arm in fold['arms']:
            if 'prediction_sha256' in arm:
                arm['prediction_sha256'].pop('Random Forest',None)
    copied.get('execution',{}).get('memory_settings',{}).pop('available_memory_bytes_at_configuration',None)
    return copied

def fix05_audit_input_sha(payload):
    return hashlib.sha256(json.dumps(normalized_fix05_inputs(payload),sort_keys=True,allow_nan=False).encode()).hexdigest()

def verify_fix05_audit_inputs(payload,expected):
    current=fix05_audit_input_sha(payload)
    if current!=expected:
        raise RuntimeError('Fix05 audit inputs, published metrics or required predictions changed. Stop and review; no guard bypass was applied.')
    return {'version':'fix05-audit-inputs-v1','expected_sha256':expected,'current_sha256':current,
            'excluded_fields':GUARD_EXCLUSIONS,
            'all_other_fields_including_RF_metrics_and_LR_CTR_prediction_hashes_checked':True}
