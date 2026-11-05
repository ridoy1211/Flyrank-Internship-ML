"""June-specific timing, candidate preservation and checkpoint integrity tests."""
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import json,sys,tempfile,unittest
from unittest.mock import patch
import duckdb
import numpy as np
import pandas as pd
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'work'),str(Path(__file__).resolve().parent)]
from research import june as j,evaluation as e,warehouse as w
from test_chronological_fix05 import synthetic_aggregate


def june_fixture_parts():
    episodes={}
    for origin in j.JUNE_TRAINING:
        raw=synthetic_aggregate(origin,n_clients=13 if origin.month==5 else 12)
        if origin.month==5:
            # A client first becomes usable in May; it was not eligible for the May episode.
            raw.loc[raw.role.eq('feature')&raw.client_hash_id.eq('synthetic_client_12'),'clicks']=0
        episodes[origin],_=e.build_episode05(raw,origin)
        if origin.month==5:may=raw.loc[raw.role.eq('outcome')].copy()
    feature=j.june_feature_from_may(may)
    asof,_=e.build_episode05(feature,j.JUNE_ORIGIN)
    return episodes,feature,asof


def fake_scores(train,test):
    assert test.decoupling_signature.isna().all()
    return {name:test.ctr_prior28.to_numpy(dtype=float)+i/100 for i,name in enumerate(e.METHODS05)},{}


def daily_june(con,path,outside=False,duplicate=False):
    dates="generate_series(DATE '2026-06-01',DATE '2026-06-30',INTERVAL 1 DAY)"
    source=f"""SELECT CAST(report_date AS DATE) AS report_date,'synthetic_client_1' AS client_hash_id,
        'synthetic_page_1' AS content_hash_id,report_date!=DATE '2026-06-05' AS gsc_data_available,
        100::BIGINT AS gsc_impressions,5::BIGINT AS gsc_clicks,200.0 AS gsc_sum_position
        FROM {dates} d(report_date)"""
    if outside:source += " UNION ALL SELECT DATE '2026-05-31','synthetic_client_1','synthetic_page_1',TRUE,100,5,200.0"
    if duplicate:source += " UNION ALL SELECT DATE '2026-06-06','synthetic_client_1','synthetic_page_1',TRUE,100,5,200.0"
    con.execute('COPY ('+source+') TO '+w.sql_literal05m(path.as_posix())+' (FORMAT PARQUET)')
    return path


class JuneValidationTests(unittest.TestCase):
    def test_dates_and_committed_protocol(self):
        self.assertEqual(e.train_origins05(j.JUNE_ORIGIN),[date(2026,m,1) for m in (3,4,5)])
        window=j.june_windows().iloc[0]
        self.assertEqual((window.start_date,window.end_date),(date(2026,6,1),date(2026,6,28)))
        path=Path(__file__).resolve().parents[1]/'work/research/protocols/june_retrospective_v2.json'
        receipt=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(j.digest_json(receipt['protocol']),j.JUNE_PROTOCOL_SHA)
        self.assertEqual(receipt['protocol']['prior_exposure']['status'],'uncertain')
        self.assertEqual(receipt['protocol']['primary_minimum_prior_clicks'],1)
        self.assertTrue(all('month=2026-06/' in x and '_sample' not in x for x in j.june_source_paths()))

    def test_unfiltered_may_reuse_retains_newly_eligible_client(self):
        episodes,feature,asof=june_fixture_parts()
        self.assertNotIn('synthetic_client_12',set(episodes[date(2026,5,1)].client_hash_id))
        self.assertIn('synthetic_client_12',set(asof.client_hash_id))
        self.assertTrue(asof.decoupling_signature.isna().all())
        self.assertEqual(set(feature.origin),{'2026-06-01'})
        with self.assertRaisesRegex(ValueError,'unfiltered'):j.june_feature_from_may(episodes[date(2026,5,1)])
        wrong=feature.copy()
        with self.assertRaisesRegex(ValueError,'Only'):j.june_feature_from_may(wrong)

    def test_arm_membership_and_frozen_hash_holdout(self):
        episodes,feature,asof=june_fixture_parts()
        plan=j.training_and_candidates_june(episodes,asof)
        known,unseen,heldout=plan
        self.assertEqual(set(unseen[2].client_hash_id),{'synthetic_client_12'})
        self.assertFalse(set(known[2].client_hash_id)&set(unseen[2].client_hash_id))
        self.assertFalse(set(heldout[1].client_hash_id)&set(heldout[2].client_hash_id))
        self.assertTrue(heldout[2].client_hash_id.map(e.heldout05).all())
        self.assertEqual(set(known[1].origin),{'2026-03-01','2026-04-01','2026-05-01'})
        altered=asof.copy();altered.loc[0,'decoupling_signature']=1
        with self.assertRaisesRegex(ValueError,'before June outcomes'):j.training_and_candidates_june(episodes,altered)

    def test_prediction_checkpoint_reuse_and_corruption_rejection(self):
        episodes,feature,asof=june_fixture_parts()
        with tempfile.TemporaryDirectory() as folder:
            con=duckdb.connect();path=Path(folder)/'predictions.parquet'
            identity={'protocol_sha256':j.JUNE_PROTOCOL_SHA,'implementation_sha256':'a'*64,
                'historical_cache_sha256':'b'*64,'dataset_revision':'synthetic_fixture_only'}
            try:
                with patch.object(j,'score_models05',side_effect=fake_scores) as scorer:
                    records,frozen,receipt=j.freeze_predictions_june(con,episodes,asof,path,identity,lambda _:None)
                    self.assertEqual(scorer.call_count,3)
                with patch.object(j,'score_models05',side_effect=AssertionError('No refit')):
                    _,again,same=j.freeze_predictions_june(con,episodes,asof,path,identity,lambda _:None)
                self.assertEqual(receipt,same)
                self.assertEqual(set(again),set(frozen))
                for arm in frozen:self.assertEqual(j.scores_sha_june(again[arm][1]),j.scores_sha_june(frozen[arm][1]))
                with path.open('ab') as stream:stream.write(b'bad bytes')
                with self.assertRaisesRegex(RuntimeError,'checksum'):
                    j.freeze_predictions_june(con,episodes,asof,path,identity,lambda _:None)
            finally:con.close()

    def test_outcome_attachment_retains_unknowns_and_score_order(self):
        episodes,feature,asof=june_fixture_parts()
        records=[];frozen={}
        for record,train,test in j.training_and_candidates_june(episodes,asof):
            scores,audit=fake_scores(train,test)
            record.update(reference_audit=audit,prediction_sha256=j.scores_sha_june(scores))
            records.append(record);frozen[record['arm']]=(test,scores)
        # Shuffle outcomes, omit an entire page, and mark another page incomplete.
        outcome=feature.copy();outcome['role']='outcome'
        key=asof.iloc[0]
        omit=outcome.client_hash_id.eq(key.client_hash_id)&outcome.content_hash_id.eq(key.content_hash_id)
        outcome=outcome.loc[~omit].sample(frac=1,random_state=13)
        outcome.loc[outcome.index[0],'available_days']=27
        protocol=json.loads((Path(__file__).resolve().parents[1]/'work/research/protocols/june_retrospective_v2.json').read_text())['protocol']
        result,paired,noise=j.report_june(episodes,feature,outcome,records,frozen,protocol)
        self.assertEqual(sum(a['evaluation']['n_page_episodes'] for a in result['folds'][0]['arms'][:2]),len(asof))
        for row in result['folds'][0]['arms']:
            self.assertEqual(row['prediction_sha256'],j.scores_sha_june(frozen[row['arm']][1]))
        self.assertGreater(sum(a['evaluation']['n_page_episodes']-a['evaluation']['n_labelable'] for a in result['folds'][0]['arms'][:2]),0)
        primary=[p for row in paired['rows'] for p in row['pairs'] if p['role']=='primary']
        self.assertEqual(len(primary),1)
        self.assertEqual(primary[0]['method_b'],'Prior CTR')
        self.assertEqual(len(noise['rows']),3)
        self.assertNotIn('client_hash_id',json.dumps(result))

    def test_june_sql_uses_only_first_28_days_and_explicit_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            con=duckdb.connect()
            try:
                source=daily_june(con,Path(folder)/'daily.parquet')
                con.register('_fix05_windows',j.june_windows())
                aggregate=con.execute(w.month_sql05m([source],'2026-06')).df()
                self.assertEqual(aggregate.raw_rows.tolist(),[28])
                self.assertEqual(aggregate.raw_days.tolist(),[28])
                self.assertEqual(aggregate.available_days.tolist(),[27])
                self.assertEqual(aggregate.impressions.tolist(),[2700])
                self.assertEqual(set(aggregate.origin),{'2026-06-01'})
            finally:con.close()

    def test_wrong_month_and_duplicate_daily_input_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            con=duckdb.connect();root=Path(folder)
            try:
                con.register('_fix05_windows',j.june_windows())
                source=daily_june(con,root/'outside.parquet',outside=True)
                with self.assertRaisesRegex(Exception,'out-of-month'):con.execute(w.month_sql05m([source],'2026-06')).df()
                source=daily_june(con,root/'duplicate.parquet',duplicate=True)
                with self.assertRaisesRegex(RuntimeError,'coverage/grain/count'):
                    w.write_checkpoint05m(con,w.month_sql05m([source],'2026-06'),root/'out.parquet',{})
            finally:con.close()

    def test_revision_change_preserves_pending_and_rejects_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);con=duckdb.connect()
            source=daily_june(con,root/'daily.parquet')
            class SecretlessConnection:
                def execute(self,query,*args):
                    return con if query.startswith('CREATE OR REPLACE SECRET') else con.execute(query,*args)
                def __getattr__(self,name):return getattr(con,name)
            identity={'dataset_revision':j.SOURCE_REVISION,'protocol_sha256':j.JUNE_PROTOCOL_SHA,'implementation_sha256':'a'*64}
            try:
                with patch.dict(j.os.environ,{'HF_TOKEN':'hf_fake'}),patch('huggingface_hub.HfApi') as api,patch.object(j,'june_source_paths',return_value=[str(source)]):
                    api.return_value.dataset_info.side_effect=[SimpleNamespace(sha=j.SOURCE_REVISION),SimpleNamespace(sha='changed')]
                    with self.assertRaisesRegex(RuntimeError,'revision changed'):
                        j.acquire_june_outcomes(SecretlessConnection(),root/'cache',identity,progress=lambda _:None)
                self.assertFalse((root/'cache/outcomes.json').exists())
                self.assertTrue(list((root/'cache/pending').glob('*.parquet')))
            finally:con.close()

    def test_retry_preserves_original_access_time_and_rejects_late_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);con=duckdb.connect()
            source=daily_june(con,root/'daily.parquet')
            identity={'dataset_revision':'synthetic_fixture_only','protocol_sha256':j.JUNE_PROTOCOL_SHA,'implementation_sha256':'a'*64}
            try:
                path,first,_=j.acquire_june_outcomes(con,root/'cache',identity,synthetic_source=source,
                    prediction_frozen_utc='2020-01-01T00:00:00+00:00',progress=lambda _:None)
                _,again,queried=j.acquire_june_outcomes(con,root/'cache',identity,
                    prediction_frozen_utc='2020-01-01T00:00:00+00:00',progress=lambda _:None)
                self.assertEqual(first,again);self.assertFalse(queried)
                with self.assertRaisesRegex(RuntimeError,'after an earlier June access'):
                    j.acquire_june_outcomes(con,root/'cache',identity,
                        prediction_frozen_utc='2100-01-01T00:00:00+00:00',progress=lambda _:None)
                with self.assertRaisesRegex(RuntimeError,'after an earlier June access'):
                    j.acquire_june_outcomes(con,root/'different-code-version',
                        {**identity,'implementation_sha256':'b'*64},synthetic_source=source,
                        prediction_frozen_utc='2100-01-01T00:00:00+00:00',progress=lambda _:None)
                self.assertFalse((root/'different-code-version/outcomes.json').exists())
            finally:con.close()


if __name__=='__main__':unittest.main()
