from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'work'))
import chronological_fix05 as c
import memory_fix05 as m


def daily_fixtures05m(con,root,clients=3,pages=12):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
    for month in c.MONTHS05:
        start,end=m.month_bounds05m(month)
        path=root/f'{month}.parquet'
        sql=f"""WITH daily AS (
          SELECT CAST(report_date AS DATE) AS report_date,client,page
          FROM generate_series(DATE '{start}',DATE '{end}'-INTERVAL 1 DAY,INTERVAL 1 DAY) t(report_date),
               range({clients}) a(client),range({pages}) b(page)
          WHERE client<12 OR report_date>=DATE '2026-03-01')
          SELECT report_date,'synthetic_client_'||client AS client_hash_id,
            'synthetic_page_'||page AS content_hash_id,
            NOT ((page%13=0 AND report_date=DATE '2026-02-05') OR
                 (page%17=0 AND report_date=DATE '2026-04-10')) AS gsc_data_available,
            CAST(100+page AS BIGINT) AS gsc_impressions,
            CAST(CASE WHEN page%4=0 THEN GREATEST(1,17-(year(report_date)-2025)*12-month(report_date))
                 ELSE 2 END AS BIGINT) AS gsc_clicks,
            CAST((100+page)*(1+page%60) AS DOUBLE) AS gsc_sum_position FROM daily"""
        con.execute('COPY ('+sql+') TO '+m.sql_literal05m(path.as_posix())+' (FORMAT PARQUET)')
        paths[month]=path
    return paths


class MemoryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.con=duckdb.connect();self.con.execute('SET threads=1')
        self.con.execute('SET preserve_insertion_order=false')
        self.paths=daily_fixtures05m(self.con,self.root/'source')
    def tearDown(self):self.con.close();self.tmp.cleanup()
    def build(self,source=None,revision=lambda:'revision-a'):
        return m.build_cache05m(self.con,c.windows05(),c.MONTHS05,self.root/'cache',
            c.protocol_hash05(),'test-implementation',source or (lambda month:[self.paths[month]]),revision,lambda x:None)

    def test_matches_original_counts_positions_coverage_and_labels(self):
        self.con.register('_fix05_windows',c.windows05())
        old=self.con.sql(c.aggregation_sql05(list(self.paths.values()))).df()
        path,_=self.build();new=self.con.sql('SELECT * FROM '+m.parquet05m([path])).df()
        keys=['origin','role']+c.KEYS05
        old=old.sort_values(keys).reset_index(drop=True);new=new.sort_values(keys).reset_index(drop=True)
        for column in old.columns:
            if column=='sum_position':np.testing.assert_allclose(old[column],new[column],rtol=1e-12)
            else:pd.testing.assert_series_equal(old[column],new[column],check_dtype=False)
        for origin in c.EPISODES05:
            a,_=c.build_episode05(old.loc[old.origin.eq(str(origin))],origin)
            b,_=c.build_episode05(new.loc[new.origin.eq(str(origin))],origin)
            pd.testing.assert_frame_equal(a,b)

    def test_cached_rerun_has_no_remote_revision_or_source_calls(self):
        path,receipt=self.build()
        impossible=Mock(side_effect=AssertionError('No source access allowed'))
        second,again=self.build(source=impossible,revision=impossible)
        self.assertEqual(path,second);self.assertEqual(receipt,again);impossible.assert_not_called()

    def test_interruption_reuses_only_completed_months(self):
        seen=[]
        def interrupted(month):
            seen.append(month)
            if month==c.MONTHS05[1]:raise RuntimeError('synthetic interruption')
            return [self.paths[month]]
        with self.assertRaisesRegex(RuntimeError,'interruption'):self.build(source=interrupted)
        self.assertTrue((self.root/'cache/months/2025-10.json').exists())
        seen=[]
        def resumed(month):seen.append(month);return [self.paths[month]]
        self.build(source=resumed)
        self.assertNotIn('2025-10',seen);self.assertEqual(len(seen),7)

    def test_revision_change_never_commits_current_month(self):
        calls=iter(['revision-a','revision-a','revision-b'])
        with self.assertRaisesRegex(RuntimeError,'revision changed'):
            self.build(revision=lambda:next(calls))
        self.assertFalse((self.root/'cache/months/2025-10.json').exists())
        self.assertTrue((self.root/'cache/pending/2025-10.parquet').exists())

    def test_wrong_partition_dates_fail(self):
        def wrong(month):return [self.paths['2025-11']] if month=='2025-10' else [self.paths[month]]
        with self.assertRaisesRegex(duckdb.InvalidInputException,'out-of-month'):self.build(source=wrong)

    def test_null_source_date_fails_inside_single_scan(self):
        path=self.root/'null_date.parquet'
        self.con.execute('COPY (SELECT * REPLACE(NULL::DATE AS report_date) FROM '+m.parquet05m([self.paths['2025-10']])+
            ' LIMIT 1) TO '+m.sql_literal05m(path.as_posix())+' (FORMAT PARQUET)')
        with self.assertRaisesRegex(duckdb.InvalidInputException,'null/out-of-month'):
            self.build(source=lambda month:[path])

    def test_duplicate_day_fails_before_trusting_checkpoint(self):
        path=self.paths['2025-10'];other=self.root/'duplicate.parquet'
        self.con.execute('COPY (SELECT * FROM '+m.parquet05m([path])+
            ' UNION ALL (SELECT * FROM '+m.parquet05m([path])+' LIMIT 1)) TO '+m.sql_literal05m(other.as_posix())+' (FORMAT PARQUET)')
        self.paths['2025-10']=other
        with self.assertRaisesRegex(RuntimeError,'grain/count'):self.build()
        self.assertFalse((self.root/'cache/months/2025-10.json').exists())

    def test_corrupt_final_checksum_is_rejected(self):
        path,_=self.build()
        with path.open('ab') as stream:stream.write(b'corruption')
        with self.assertRaisesRegex(RuntimeError,'checksum'):self.build()

    def test_empty_source_adds_no_zero_filled_pages(self):
        path=self.root/'empty.parquet'
        self.con.execute('COPY (SELECT * FROM '+m.parquet05m([self.paths['2025-10']])+
            ' WHERE FALSE) TO '+m.sql_literal05m(path.as_posix())+' (FORMAT PARQUET)')
        self.con.register('_fix05_windows',c.windows05())
        result=self.con.sql(m.month_sql05m([path])).df()
        self.assertEqual(len(result),0)

    def test_masks_union_detects_overlap_between_chunks(self):
        self.con.register('_fix05_windows',c.windows05())
        partial=self.root/'partial.parquet'
        m.write_checkpoint05m(self.con,m.month_sql05m([self.paths['2025-10']]),partial,{'test':'mask'})
        with self.assertRaisesRegex(RuntimeError,'grain/count'):
            m.write_checkpoint05m(self.con,m.merge_sql05m([partial,partial],'2025-11-01','feature'),
                self.root/'bad_merge.parquet',{'test':'merge'},merged=True)


if __name__=='__main__':unittest.main(verbosity=2)
