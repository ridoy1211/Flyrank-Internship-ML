from pathlib import Path
import importlib.util,json,unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('diagnostics04',ROOT/'work/diagnostics_fix04.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(n=300):
    return pd.DataFrame({'client_hash_id':np.repeat(['private_a','private_b','private_c'],n//3),
        'clicks_prior28':np.tile([1,6,25],n//3),
        'decoupling_signature':np.arange(n)%2})

class FrozenScoreDiagnosticsChecks(unittest.TestCase):
    def test_declared_budget_is_not_silently_reduced(self):
        r=m.diagnostic_budget([1.,0.],[1,0],k=50)
        self.assertIsNone(r['precision']);self.assertEqual(r['n_pages'],2)
        self.assertIn('not silently reduced',r['reason'])

    def test_constant_ties_follow_canonical_input_order(self):
        r=m.diagnostic_budget(np.zeros(100),[0]*50+[1]*50)
        self.assertEqual(r['hits'],0)

    def test_bad_score_lengths_and_values_fail(self):
        with self.assertRaises(ValueError):m.diagnostic_budget([1.],[0,1])
        with self.assertRaises(ValueError):m.diagnostic_budget([np.nan],[1])

    def test_support_restrictions_do_not_change_labels_or_scores(self):
        d=fixture();before=d.copy(deep=True);s=np.arange(300,dtype=float);copy=s.copy()
        rows=m.support_sensitivity(d,{'Logistic Regression':s})
        self.assertEqual([r['n_pages'] for r in rows],[300,200,100])
        self.assertEqual([r['minimum_prior_clicks'] for r in rows],[1,5,20])
        pd.testing.assert_frame_equal(d,before);np.testing.assert_array_equal(s,copy)

    def test_identical_rankings_overlap_completely(self):
        d=fixture();s=np.arange(300,dtype=float)
        row=m.ranking_overlap(d,{'Logistic Regression':s,'Prior CTR':s.copy()})[0]
        self.assertEqual(row['overlap_pages'],50);self.assertEqual(row['jaccard'],1.)
        self.assertAlmostEqual(row['spearman_rank_correlation_all_test_pages'],1.)
        self.assertEqual(row['reference_only_positive_count'],0)

    def test_reversed_and_constant_rankings(self):
        d=fixture();s=np.arange(300,dtype=float)
        rows=m.ranking_overlap(d,{'Logistic Regression':s,'reverse':-s,'constant':np.zeros(300)})
        self.assertEqual(rows[0]['overlap_pages'],0)
        self.assertAlmostEqual(rows[0]['spearman_rank_correlation_all_test_pages'],-1.)
        self.assertIsNone(rows[1]['spearman_rank_correlation_all_test_pages'])

    def test_small_clients_have_explicit_effective_budget(self):
        d=pd.DataFrame({'client_hash_id':['private_large']*50+['private_small']*5,
                        'decoupling_signature':[0]*50+[1]*5})
        out=m.client_quota_diagnostics(d,{'method':np.arange(55,dtype=float)})
        summary=out['secondary_summary'][0]
        self.assertEqual(summary['total_reviewed_pages'],15)
        self.assertEqual(summary['total_positive_hits'],5)
        self.assertAlmostEqual(summary['macro_client_precision'],.5)
        self.assertAlmostEqual(summary['pooled_quota_precision'],1/3)
        self.assertFalse(summary['all_clients_had_full_quota'])
        self.assertNotIn('private_large',str(out));self.assertNotIn('private_small',str(out))

    def test_full_diagnostics_are_json_serializable_and_preserve_primary_policy(self):
        d=fixture();s=np.arange(300,dtype=float)
        out=m.run_fixed_diagnostics(d,{'Logistic Regression':s,'Prior CTR':s.copy()})
        json.dumps(out,allow_nan=False)
        self.assertEqual(out['interpretation']['primary_global_budget'],50)
        self.assertEqual(out['interpretation']['primary_support_minimum_prior_clicks'],1)
        self.assertEqual(len(out['support_sensitivity']),6)

    def test_continuation_embeds_the_tested_helper(self):
        source=(ROOT/'work/diagnostics_fix04_continuation.py').read_text(encoding='utf-8')
        helper=(ROOT/'work/diagnostics_fix04.py').read_text(encoding='utf-8')
        self.assertTrue(source.startswith(helper));self.assertNotIn('from __future__',source)
        compile(source,'continuation04','exec')

if __name__=='__main__':unittest.main(verbosity=2)
