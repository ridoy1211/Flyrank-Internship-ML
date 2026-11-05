from pathlib import Path
import importlib.util, json, unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('measurement',ROOT/'work/measurement_fix02.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def inputs():
    keys=pd.DataFrame({'client_hash_id':['a','b','c','d'],
                       'content_hash_id':['shared','shared','p3','p4']})
    f=keys.copy()
    f['raw_rows']=28;f['raw_days']=28;f['available_days_prior28']=28;f['bad_counts']=0
    f['impressions_prior28']=1000;f['clicks_prior28']=[100,0,100,100]
    f['avg_position_prior28']=5.
    r=keys.copy()
    r['raw_rows']=31;r['raw_days']=31;r['available_days_outcome28']=[28,28,27,28]
    r['available_days_outcome31']=[31,31,30,31];r['bad_counts']=0
    r['impressions_outcome28']=1000;r['clicks_outcome28']=80
    r['impressions_outcome31']=1107;r['clicks_outcome31']=89
    meta=keys.copy();meta['content_type']='a';meta['main_intent']='information'
    return f,r,meta

class MeasurementChecks(unittest.TestCase):
    def test_exact_boundaries(self):
        p=pd.Series([1000]*6);c=pd.Series([100]*6)
        i=pd.Series([900,1100,899,1101,1000,1000]);o=pd.Series([85,85,85,85,86,84])
        y=m.repaired_label(p,c,i,o,pd.Series([True]*6))
        self.assertEqual(y.tolist(),[1,1,0,0,0,1])

    def test_zero_prior_clicks_and_missing_outcomes_stay_unknown(self):
        y=m.repaired_label(pd.Series([1000]*3),pd.Series([0,10,10]),
            pd.Series([1000,1000,1000]),pd.Series([0,0,0]),pd.Series([True,False,True]))
        self.assertTrue(pd.isna(y.iloc[0]));self.assertTrue(pd.isna(y.iloc[1]))
        self.assertEqual(y.iloc[2],1)

    def test_full_prior_coverage_is_required(self):
        f,r,meta=inputs();f.loc[0,'available_days_prior28']=27
        d,a=m.prepare_equal_windows(f,r,meta)
        self.assertEqual(len(d),1)
        self.assertEqual(a['n_as_of_eligible'],2)

    def test_missing_followup_is_counted_not_zero_filled(self):
        f,r,meta=inputs();r=r.iloc[1:]
        d,a=m.prepare_equal_windows(f,r,meta)
        self.assertEqual(a['n_as_of_eligible'],3)
        self.assertEqual(a['n_eligible_without_complete_outcome'],2)
        self.assertEqual(len(d),1)

    def test_compound_page_key_prevents_cross_client_duplication(self):
        f,r,meta=inputs();d,a=m.prepare_equal_windows(f,r,meta)
        self.assertEqual(len(d),2)
        self.assertEqual(a['n_zero_prior_clicks_with_full_prior_exposure'],1)
        self.assertEqual(a['paired_calendar_comparison_n'],2)
        self.assertEqual(sum(x['n_pages'] for x in a['paired_label_counts']),2)
        cell=next(x for x in a['paired_label_counts'] if x['equal28_label']==1 and x['unequal31_label']==0)
        self.assertEqual(cell['n_pages'],2)

    def test_duplicate_days_and_keys_fail(self):
        f,r,meta=inputs();f.loc[0,'raw_rows']=29
        with self.assertRaisesRegex(ValueError,'duplicate page-day'):
            m.prepare_equal_windows(f,r,meta)
        f,r,meta=inputs();meta=pd.concat([meta,meta.iloc[:1]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'duplicate page keys'):
            m.prepare_equal_windows(f,r,meta)

    def test_bad_and_fractional_counts_fail(self):
        f,r,meta=inputs();r.loc[0,'bad_counts']=1
        with self.assertRaisesRegex(ValueError,'null/negative'):
            m.prepare_equal_windows(f,r,meta)
        with self.assertRaisesRegex(ValueError,'integers'):
            m.repaired_label(pd.Series([1000]),pd.Series([10]),pd.Series([1000]),
                             pd.Series([2.5]),pd.Series([True]))

    def test_notebook_sources_compile_and_embed_current_measurement(self):
        nb=json.loads((ROOT/'work/research/archive/fix02_equal_windows.ipynb').read_text(encoding='utf-8'))
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec')
        self.assertEqual(''.join(nb['cells'][2]['source']),
                         (ROOT/'work/measurement_fix02.py').read_text(encoding='utf-8'))
        old=(ROOT/'work/model_comparison.py').read_text(encoding='utf-8')
        expected=old.replace('prior30','prior28').replace('Stage 1: compare all candidates','Stage 2: compare all candidates').replace(
            'The original February/March outcome is deliberately retained by the calling notebook.\nResults remain exploratory until the separate outcome and temporal-validation repairs.',
            'The calling notebook supplies the repaired equal-window, eligible-page outcome.\nResults remain exploratory until the separate temporal-validation and uncertainty repairs.')
        self.assertEqual(''.join(nb['cells'][3]['source']),expected)
        self.assertIn('Restart the Colab runtime',''.join(nb['cells'][1]['source']))

if __name__=='__main__':unittest.main(verbosity=2)
