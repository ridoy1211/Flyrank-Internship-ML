import json
from datetime import date,timedelta
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'work'))
import chronological_fix05 as c


def synthetic_aggregate(origin,n_clients=12,pages=60):
    rng=np.random.default_rng(origin.toordinal());rows=[]
    for client in range(n_clients):
        for page in range(pages):
            impressions=int(rng.integers(500,5000)); clicks=int(rng.integers(1,100))
            positive=(page+client)%4==0
            for role in ('feature','outcome'):
                available=24 if role=='outcome' and page%13==0 else 28
                rows.append({'origin':str(origin),'role':role,'client_hash_id':f'synthetic_client_{client}',
                    'content_hash_id':f'synthetic_page_{page}','raw_rows':28,'raw_days':28,
                    'available_days':available,'bad_counts':0,'bad_position':0,
                    'impressions':impressions,'clicks':clicks if role=='feature' else (clicks*4//5 if positive else clicks),
                    'sum_position':impressions*float(rng.uniform(1,60))})
    return pd.DataFrame(rows)


class ChronologicalTests(unittest.TestCase):
    def test_dates_and_label_maturity(self):
        windows=c.windows05()
        self.assertTrue(((pd.to_datetime(windows.end_date)-pd.to_datetime(windows.start_date)).dt.days==27).all())
        self.assertEqual(c.train_origins05(date(2026,3,1)),[date(2025,11,1),date(2025,12,1),date(2026,1,1)])
        self.assertEqual(c.train_origins05(date(2026,4,1)),[date(2026,m,1) for m in (1,2,3)])
        self.assertEqual(c.train_origins05(date(2026,5,1)),[date(2026,m,1) for m in (1,2,3)])
        for t in c.TEST_ORIGINS05:
            self.assertEqual(len(c.train_origins05(t)),3)
            self.assertTrue(all(s+timedelta(days=27)<=t-timedelta(days=4) for s in c.train_origins05(t)))
        self.assertFalse(any(m>='2026-06' for m in c.MONTHS05))

    def test_exact_boundary_and_unknown_retention(self):
        a=synthetic_aggregate(date(2026,2,1),1,4)
        a['impressions']=100;a['clicks']=20
        outcome=a.role.eq('outcome')
        a.loc[outcome,'clicks']=[17,17,18,17]
        a.loc[outcome,'impressions']=[90,110,100,89]
        a.loc[outcome,'available_days']=28
        a.loc[outcome&a.content_hash_id.eq('synthetic_page_1'),'available_days']=27
        d,audit=c.build_episode05(a,date(2026,2,1))
        self.assertEqual(len(d),4)
        self.assertEqual(d.decoupling_signature.tolist(),[1,pd.NA,0,0])
        self.assertEqual(audit['n_unknown'],1)

    def test_sparse_features_excluded_without_zero_fill(self):
        a=synthetic_aggregate(date(2026,2,1),1,3)
        a.loc[a.role.eq('feature')&a.content_hash_id.eq('synthetic_page_1'),'available_days']=27
        a.loc[a.role.eq('feature')&a.content_hash_id.eq('synthetic_page_2'),'clicks']=0
        d,_=c.build_episode05(a,date(2026,2,1))
        self.assertEqual(d.content_hash_id.tolist(),['synthetic_page_0'])
        self.assertTrue(d.decoupling_signature.isna().all())

    def test_duplicate_and_bad_count_fail_closed(self):
        a=synthetic_aggregate(date(2026,2,1),1,3)
        a.loc[0,'raw_rows']=29
        with self.assertRaisesRegex(ValueError,'Duplicate'):c.build_episode05(a,date(2026,2,1))
        a.loc[0,'raw_rows']=28;a.loc[0,'bad_counts']=1
        with self.assertRaisesRegex(ValueError,'count data'):c.build_episode05(a,date(2026,2,1))

    def test_rank_before_filter_and_missing_bounds(self):
        d=pd.DataFrame({'decoupling_signature':pd.array([pd.NA,0,1,1],dtype='Int64')})
        m=c.metrics05(d,[4,3,2,1],ks=(2,5))
        q=m['queues'][0]
        self.assertEqual(q['observed_hits'],0);self.assertEqual(q['unknown_outcomes'],1)
        self.assertIsNone(q['precision_if_complete'])
        self.assertEqual((q['precision_lower_bound'],q['precision_upper_bound']),(0,.5))
        self.assertEqual(m['complete_case_rank_metrics'][0]['precision'],.5)
        self.assertEqual(m['queues'][1]['reason'],'insufficient_as_of_pages')

    def test_tie_ranges_cover_outcomes_and_missing(self):
        d=pd.DataFrame({'decoupling_signature':pd.array([1,pd.NA,0,1],dtype='Int64')})
        q=c.metrics05(d,[1,1,1,1],ks=(2,))['queues'][0]
        self.assertEqual(q['cutoff_tie_count'],4)
        self.assertEqual(q['tie_and_missing_lower_bound'],0)
        self.assertEqual(q['tie_and_missing_upper_bound'],1)

    def test_hash_and_fold_membership_without_future_labels(self):
        episodes={};audits=[]
        for origin in c.EPISODES05:
            episodes[origin],a=c.build_episode05(synthetic_aggregate(origin),origin);audits.append(a)
        calls=[]
        def fake_score(train,test):
            calls.append((train.copy(),test.copy()))
            return {name:test.ctr_prior28.to_numpy(dtype=float) for name in c.METHODS05},{}
        with patch.object(c,'score_models05',side_effect=fake_score):
            result=c.run_chronological05(episodes,audits)
        for fold in result['folds']:
            for arm in fold['arms']:
                if arm['status']=='scored':
                    self.assertEqual(len(arm['metrics']),24)
                    self.assertEqual(len(arm['prediction_sha256']),8)
        for train,test in calls:
            test_origin=date.fromisoformat(test.origin.iloc[0])
            self.assertTrue(set(train.origin).issubset({str(s) for s in c.train_origins05(test_origin)}))
            self.assertFalse(train.decoupling_signature.isna().any())
        # Both positive and negative deterministic assignments exist; not tuned to labels.
        self.assertEqual([c.heldout05('a')]*3,[c.heldout05('a') for _ in range(3)])
        self.assertTrue(any(c.heldout05(f'synthetic_client_{i}') for i in range(12)))
        payload=json.dumps(result,allow_nan=False)
        self.assertNotIn('synthetic_client_',payload)
        self.assertNotIn('synthetic_page_',payload)

    def test_actual_fits_ignore_future_outcomes(self):
        train,_=c.build_episode05(synthetic_aggregate(date(2026,1,1),4,30),date(2026,1,1))
        train=train.loc[train.decoupling_signature.notna()]
        test,_=c.build_episode05(synthetic_aggregate(date(2026,2,1),2,30),date(2026,2,1))
        a,_=c.score_models05(train,test)
        test.decoupling_signature=1
        test.impressions_outcome=10**9
        b,_=c.score_models05(train,test)
        self.assertEqual(list(a),c.METHODS05)
        for name in a:np.testing.assert_array_equal(a[name],b[name])

    def test_no_silent_shorter_labelled_history(self):
        episodes={};audits=[]
        for origin in c.EPISODES05:
            episodes[origin],audit=c.build_episode05(synthetic_aggregate(origin),origin);audits.append(audit)
        episodes[date(2025,12,1)]['decoupling_signature']=pd.array([pd.NA]*len(episodes[date(2025,12,1)]),dtype='Int64')
        with patch.object(c,'score_models05') as scorer:
            # Check the first two planned folds only; later folds have a different valid history.
            with patch.object(c,'TEST_ORIGINS05',c.TEST_ORIGINS05[:2]):
                results=c.run_chronological05(episodes,audits)
        self.assertFalse(scorer.called)
        self.assertEqual([f['status'] for f in results['folds']],['insufficient_history']*2)

    def test_warehouse_sql_on_daily_parquet(self):
        import duckdb
        w=c.windows05(); w=w.loc[w.origin.eq('2026-02-01')]
        rows=[]
        for start in [date(2026,1,1),date(2026,2,1)]:
            for i in range(28):
                rows.append({'report_date':start+timedelta(days=i),'client_hash_id':'synthetic_client',
                    'content_hash_id':'synthetic_page','gsc_data_available':i!=0,
                    'gsc_impressions':10,'gsc_clicks':1,'gsc_sum_position':30.0})
        con=duckdb.connect();con.register('_fix05_windows',w);con.register('fixture',pd.DataFrame(rows))
        with tempfile.TemporaryDirectory() as tmp:
            path=(Path(tmp)/'daily.parquet').as_posix()
            con.execute("COPY fixture TO '"+path+"' (FORMAT PARQUET)")
            a=con.sql(c.aggregation_sql05([path])).df()
        con.close()
        self.assertEqual(len(a),2)
        self.assertEqual(a.available_days.tolist(),[27,27])
        self.assertEqual(a.impressions.tolist(),[270,270])
        self.assertEqual(a.clicks.tolist(),[27,27])
        self.assertEqual(a.sum_position.tolist(),[810.,810.])


if __name__=='__main__':unittest.main(verbosity=2)
