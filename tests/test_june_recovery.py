"""Check candidate-scoped recovery without using warehouse data."""
from pathlib import Path
from datetime import date
import json,sys,tempfile,unittest
from unittest.mock import patch
import duckdb
import pandas as pd
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'work'),str(Path(__file__).resolve().parent)]
from research import june_recovery as r,june as j,evaluation as e,warehouse as w
from test_chronological_fix05 import synthetic_aggregate


class JuneRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.con=duckdb.connect();self.extractor='e'*64;self.recovery='f'*64
        rows=[]
        for origin in e.EPISODES05:
            a=synthetic_aggregate(origin,n_clients=13 if origin.month==5 else 12)
            if origin.month==5:
                a.loc[a.role.eq('feature')&a.client_hash_id.eq('synthetic_client_12'),'clicks']=0
            a['row_mask']=2**28-1
            a['partial_day_total']=a.raw_days
            a['available_mask']=a.available_days.map(lambda n:2**int(n)-1)
            rows.append(a)
        history=pd.concat(rows,ignore_index=True);self.con.register('_history_test',history)
        self.cache=self.root/'windows.parquet'
        w.write_checkpoint05m(self.con,'SELECT * FROM _history_test',self.cache,
            {'protocol_sha256':e.protocol_hash05(),'source_months':e.MONTHS05,
             'kind':'combined_verified_windows','dataset_revision':'synthetic_fixture_only',
             'implementation_sha256':self.extractor},merged=True)
        self.protocol=json.loads((Path(__file__).resolve().parents[1]/'work/research/protocols/june_retrospective_v2.json').read_text())
        self.source=self.root/'daily.parquet';self.make_daily(self.source,False)

    def tearDown(self):
        self.con.close();self.temp.cleanup()

    def make_daily(self,path,inside):
        daily="""SELECT DATE '2026-06-01'+CAST(d AS INT) AS report_date,
            'synthetic_client_'||c AS client_hash_id,'synthetic_page_'||p AS content_hash_id,
            NOT(p%13=0 AND d=5) AS gsc_data_available,1000::BIGINT AS gsc_impressions,
            CASE WHEN p%4=0 THEN 0 ELSE 5 END::BIGINT AS gsc_clicks,20000.0 AS gsc_sum_position
            FROM range(13) c(c) CROSS JOIN range(60) p(p) CROSS JOIN range(28) d(d)
            WHERE NOT(c=0 AND p=1)"""
        client='synthetic_client_0' if inside else 'outside_candidate'
        daily+=" UNION ALL SELECT DATE '2026-06-02','"+client+"','synthetic_page_2',TRUE,1000,5,20000.0"
        if not inside:daily+=" UNION ALL SELECT DATE '2026-06-02','outside_candidate','synthetic_page_2',TRUE,1000,5,20000.0"
        self.con.execute('COPY ('+daily+') TO '+w.sql_literal05m(path.as_posix())+' (FORMAT PARQUET)')

    def run_fixture(self,source=None):
        return r.rerun_june_after_runtime_reset(self.protocol,self.extractor,self.recovery,
            history_cache=self.cache,synthetic=True,synthetic_source=source or self.source,
            output_root=self.root/'results',private_root=self.root/'private',progress=lambda _:None)

    def test_outside_duplicates_preserved_candidates_and_retry(self):
        run=self.run_fixture();result=json.loads((run/'june_validation.json').read_text())
        ex=result['execution'];audit=ex['june_source']
        self.assertEqual(audit['global_source_quality']['duplicate_daily_key_groups'],1)
        self.assertEqual(audit['candidate_source_quality']['n_failed_groups'],0)
        self.assertEqual(audit['candidate_pages_removed'],0)
        self.assertFalse(ex['original_prediction_checkpoint_recovered'])
        self.assertEqual(ex['model_fits_this_run'],3)
        self.assertTrue(ex['previous_June_source_inspection_disclosed'])
        for arm in result['folds'][0]['arms']:
            self.assertEqual(arm['n_as_of_candidates'],arm['evaluation']['n_page_episodes'])
        self.assertGreater(sum(a['evaluation']['n_page_episodes']-a['evaluation']['n_labelable'] for a in result['folds'][0]['arms'][:2]),0)
        source_sha=w.file_hash05m(self.source)
        with patch.object(j,'score_models05',side_effect=AssertionError('No refit')):
            retry=self.run_fixture()
        again=json.loads((retry/'june_validation.json').read_text())
        self.assertEqual(again['folds'],result['folds'])
        self.assertEqual(again['execution']['model_fits_this_run'],0)
        self.assertEqual(w.file_hash05m(self.source),source_sha)
        self.assertEqual(again['execution']['june_source']['global_failed_aggregate_sha256'],audit['global_failed_aggregate_sha256'])
        for file in run.glob('*.json'):
            self.assertNotIn('synthetic_client_',file.read_text())

    def test_inside_duplicate_stops_before_metrics(self):
        path=self.root/'inside.parquet';self.make_daily(path,True)
        with self.assertRaisesRegex(RuntimeError,'overlaps the frozen'):
            self.run_fixture(path)
        self.assertFalse(list((self.root/'results').rglob('june_validation.json')))

    def test_changed_protocol_stops(self):
        self.protocol['protocol']['primary_minimum_prior_clicks']=20
        with self.assertRaisesRegex(RuntimeError,'protocol'):
            self.run_fixture()

    def test_changed_frozen_features_stop(self):
        self.run_fixture()
        private=self.root/'private'/self.recovery[:16]
        episodes={}
        for origin in j.JUNE_TRAINING:
            a=self.con.execute('SELECT * FROM '+w.parquet05m([self.cache])+' WHERE origin=?',[str(origin)]).df()
            episodes[origin],_=e.build_episode05(a,origin)
        may=self.con.execute('SELECT * FROM '+w.parquet05m([self.cache])+" WHERE origin='2026-05-01' AND role='outcome'").df()
        asof,_=e.build_episode05(j.june_feature_from_may(may),j.JUNE_ORIGIN)
        asof.loc[0,'avg_position_prior28']+=1
        with self.assertRaisesRegex(RuntimeError,'differs from the full May-derived'):
            r.scope_cached_june_outcomes(self.con,private/'global_june_aggregate.parquet',
                private/'predictions.parquet',asof,self.root/'forbidden.parquet',{})
        self.assertFalse((self.root/'forbidden.parquet').exists())


if __name__=='__main__':unittest.main()
